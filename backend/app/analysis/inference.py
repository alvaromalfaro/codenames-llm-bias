"""Generic cluster-bootstrap inference for the metrics.

Every metric in this package is a ratio computed over observations that are **not independent**:
turns inside a game share a board, a keycard and an opponent. Treating the thousands of card- or
turn-level observations as independent would produce intervals far narrower than the data supports.
So the resampling unit here is the **game**, and everything else follows from that.

The layer is deliberately metric-agnostic. It knows nothing about CIT, IAE or conc-SD: a caller
supplies a list of cluster ids and an ``estimator`` closure that recomputes its whole metric over a
subset of those clusters and returns named scalar cells. Adding a metric means writing an adapter,
never touching this file.

Three properties are load-bearing and are asserted by the tests:

  * **the estimator must arrive with its cuts already frozen.** Any banding or quantile boundary has
    to be computed once over the full data and captured in the closure. Recomputing tercile cuts
    inside a replicate would let the bands drift with the resample, so a "band 1" interval would be
    an average over shifting definitions rather than an interval for a fixed quantity;
  * **contrasts are paired.** ``probe_all - control_all`` is accumulated *within* each replicate from
    the same draw, so the difference distribution carries the correlation between the two cells.
    Differencing two independently-computed intervals throws that correlation away and inflates the
    width - usually enough to hide a real difference;
  * **degenerate replicates are dropped and counted, never imputed.** A thin band can yield no
    comparable pairs on some resamples. Such a replicate contributes nothing to that cell's
    percentiles, and the drop count is reported so a CI resting on a minority of replicates is
    visible as such.

No p-values and no hypothesis tests. Each cell reports an effect, a percentile interval, and a
minimum detectable effect as an a-priori-style sensitivity bound.

Pure computation: this module touches no database and holds no state between calls.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np

logger = logging.getLogger(__name__)

# scipy is not a dependency of this project, so the two normal quantiles are constants rather than
# calls to ``scipy.stats.norm.ppf``. They are standard and fixed by the preregistered alpha/power.
# z_{0.975}, alpha = 0.05 two-sided
Z_ALPHA_TWO_SIDED: float = 1.959963984540054
Z_POWER: float = 0.841621233572914  # z_{0.80}, power = 0.8
MDE_MULTIPLIER: float = Z_ALPHA_TWO_SIDED + Z_POWER  # ~2.8016

DEFAULT_REPLICATES: int = 2000
DEFAULT_SEED: int = 2026
DEFAULT_MAX_DROPPED_FRACTION: float = 0.20
DEFAULT_CI_PERCENTILES: tuple[float, float] = (2.5, 97.5)

# Generic in the cluster-id type: callables are contravariant in their arguments, so an estimator
# written for game-id strings is not a Callable[[Sequence[Hashable]], ...].
ClusterId = TypeVar("ClusterId", bound=Hashable)
Estimator = Callable[[Sequence[ClusterId]], Mapping[str, float]]
ContrastSpec = tuple[str, str, str]  # (contrast name, cell a, cell b)


def minimum_detectable_effect(standard_error: float) -> float:
    """The smallest true deviation from the null detectable at alpha 0.05, power 0.8.

    ``(z_{1-alpha/2} + z_{1-beta}) * SE``. This is a sensitivity bound, not a test: it says what the
    design could have found, which is the honest thing to report next to an interval that contains
    the null.
    """
    return MDE_MULTIPLIER * standard_error


@dataclass(frozen=True)
class CellEstimate:
    """One named scalar: its full-data point estimate, bootstrap interval and sensitivity."""

    name: str
    point: float | None
    ci_low: float | None
    ci_high: float | None
    standard_error: float | None
    n_used: int
    n_dropped: int
    dropped_fraction: float
    reliable: bool
    mde: float | None
    null_value: float | None = None

    @property
    def excludes_null(self) -> bool | None:
        """Whether the interval excludes the null. ``None`` when either is undefined.

        A convenience for reporting, not a test - there is no p-value attached to it.
        """
        if self.null_value is None or self.ci_low is None or self.ci_high is None:
            return None
        return self.null_value < self.ci_low or self.null_value > self.ci_high


@dataclass(frozen=True)
class ContrastEstimate:
    """A paired difference between two cells, accumulated within each replicate."""

    name: str
    cell_a: str
    cell_b: str
    point: float | None
    ci_low: float | None
    ci_high: float | None
    standard_error: float | None
    n_used: int
    n_dropped: int
    reliable: bool

    @property
    def excludes_zero(self) -> bool | None:
        if self.ci_low is None or self.ci_high is None:
            return None
        return self.ci_low > 0.0 or self.ci_high < 0.0


@dataclass(frozen=True)
class BootstrapResult:
    n_clusters: int
    n_replicates: int
    seed: int
    cells: Mapping[str, CellEstimate]
    contrasts: Mapping[str, ContrastEstimate]
    elapsed_seconds: float


def _is_usable(value: float | None) -> bool:
    """A cell value is usable unless it is missing or non-finite (NaN / inf)."""
    return value is not None and math.isfinite(value)


def _summarise(
    values: list[float],
    n_replicates: int,
    *,
    ci_percentiles: tuple[float, float],
    max_dropped_fraction: float,
) -> tuple[float | None, float | None, float | None, int, int, float, bool]:
    """Percentile interval and SE over the usable replicates only."""
    n_used = len(values)
    n_dropped = n_replicates - n_used
    dropped_fraction = n_dropped / n_replicates if n_replicates else 1.0
    reliable = dropped_fraction <= max_dropped_fraction and n_used >= 2
    if n_used < 2:
        return None, None, None, n_used, n_dropped, dropped_fraction, False
    array = np.asarray(values, dtype=np.float64)
    low, high = np.percentile(array, list(ci_percentiles))
    # ddof=1: the replicates are a sample, and the bootstrap SD estimates the sampling SD.
    standard_error = float(np.std(array, ddof=1))
    return (
        float(low),
        float(high),
        standard_error,
        n_used,
        n_dropped,
        dropped_fraction,
        reliable,
    )


def cluster_bootstrap(
    cluster_ids: Sequence[ClusterId],
    estimator: Estimator[ClusterId],
    *,
    n_replicates: int = DEFAULT_REPLICATES,
    seed: int = DEFAULT_SEED,
    contrasts: Sequence[ContrastSpec] = (),
    max_dropped_fraction: float = DEFAULT_MAX_DROPPED_FRACTION,
    ci_percentiles: tuple[float, float] = DEFAULT_CI_PERCENTILES,
    null_value: float | None = None,
) -> BootstrapResult:
    """Percentile cluster bootstrap over ``cluster_ids``, resampled with replacement.

    ``estimator`` recomputes the entire metric over the clusters it is handed and returns named
    scalar cells. It must already hold any quantile cuts frozen from the full data - this function
    deliberately offers no hook to recompute them per replicate.

    Point estimates come from a single call on the **full** cluster list; the replicates supply
    dispersion only, so the reported centre is the estimate the metric actually produced rather than
    a bootstrap mean shifted by resampling bias.
    """
    if not cluster_ids:
        raise ValueError("cluster_bootstrap needs at least one cluster")
    if n_replicates < 1:
        raise ValueError("n_replicates must be positive")

    started = time.perf_counter()
    point_cells = dict(estimator(list(cluster_ids)))

    rng = np.random.default_rng(seed)
    size = len(cluster_ids)
    index = np.arange(size)

    cell_samples: dict[str, list[float]] = {name: [] for name in point_cells}
    contrast_samples: dict[str, list[float]] = {
        spec[0]: [] for spec in contrasts}

    for _ in range(n_replicates):
        drawn = rng.choice(index, size=size, replace=True)
        subset = [cluster_ids[position] for position in drawn]
        replicate = estimator(subset)

        for name in cell_samples:
            value = replicate.get(name)
            if _is_usable(value):
                # type: ignore[arg-type]
                cell_samples[name].append(float(value))

        for name, cell_a, cell_b in contrasts:
            value_a = replicate.get(cell_a)
            value_b = replicate.get(cell_b)
            # Both terms come from THIS draw, so the difference is paired.
            if _is_usable(value_a) and _is_usable(value_b):
                contrast_samples[name].append(
                    float(value_a) - float(value_b))  # type: ignore[arg-type]

    cells: dict[str, CellEstimate] = {}
    for name, values in cell_samples.items():
        low, high, se, n_used, n_dropped, dropped_fraction, reliable = _summarise(
            values,
            n_replicates,
            ci_percentiles=ci_percentiles,
            max_dropped_fraction=max_dropped_fraction,
        )
        raw_point = point_cells.get(name)
        point = float(raw_point) if raw_point is not None and math.isfinite(
            raw_point) else None
        cells[name] = CellEstimate(
            name=name,
            point=point,
            ci_low=low,
            ci_high=high,
            standard_error=se,
            n_used=n_used,
            n_dropped=n_dropped,
            dropped_fraction=dropped_fraction,
            reliable=reliable,
            mde=minimum_detectable_effect(se) if se is not None else None,
            null_value=null_value,
        )

    contrast_results: dict[str, ContrastEstimate] = {}
    for name, cell_a, cell_b in contrasts:
        low, high, se, n_used, n_dropped, _fraction, reliable = _summarise(
            contrast_samples[name],
            n_replicates,
            ci_percentiles=ci_percentiles,
            max_dropped_fraction=max_dropped_fraction,
        )
        value_a = point_cells.get(cell_a)
        value_b = point_cells.get(cell_b)
        point = (
            float(value_a) - float(value_b)
            if value_a is not None
            and value_b is not None
            and math.isfinite(value_a)
            and math.isfinite(value_b)
            else None
        )
        contrast_results[name] = ContrastEstimate(
            name=name,
            cell_a=cell_a,
            cell_b=cell_b,
            point=point,
            ci_low=low,
            ci_high=high,
            standard_error=se,
            n_used=n_used,
            n_dropped=n_dropped,
            reliable=reliable,
        )

    elapsed = time.perf_counter() - started
    logger.info(
        "cluster bootstrap: %d clusters, %d replicates, %d cells in %.1fs",
        size,
        n_replicates,
        len(cells),
        elapsed,
    )
    return BootstrapResult(
        n_clusters=size,
        n_replicates=n_replicates,
        seed=seed,
        cells=cells,
        contrasts=contrast_results,
        elapsed_seconds=elapsed,
    )
