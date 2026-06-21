"""Interactive semi-automatic CLI. Exposes main().

Orchestrates pool loading, covariate balancing, lexical composition, role assignment, the
semi-automatic dilemma loop (the manual target/bridge selections stay manual), position
randomization and serialization. Determinism: every random draw derives from the recorded board seed.
"""

from __future__ import annotations


def main() -> None:
    """Run the interactive board-bank generation flow (manual steps stay manual)."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
