# Glasgow Norms

`glasgow_norms.csv` holds the Glasgow Norms: ratings of 5,553 English words on nine psycholinguistic
scales. The generator takes the **gender** scale as the load of a card: it chooses the loaded cards
and their pole with it (decision D-07). The other eight scales are covariates.

## Source

Electronic supplementary material 2 (`13428_2018_1099_MOESM2_ESM.csv`) of:

> Scott, G. G., Keitel, A., Becirspahic, M., Yao, B., & Sereno, S. C. (2019). The Glasgow Norms:
> Ratings of 5,500 words on nine scales. *Behavior Research Methods*, 51(3), 1258–1270.
> <https://doi.org/10.3758/s13428-018-1099-3>

Downloaded on 2026-10-28 from
<https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-018-1099-3/MediaObjects/13428_2018_1099_MOESM2_ESM.csv>.

* **SHA-256:** `30d7776eed9219767f8a6aabdf95d7120cd050ddeab431acef13d435807bbc39`
* **Changes from the original:** none; only the file name. The file keeps its CRLF line endings, and
  `.gitattributes` stores it byte for byte (`-text`), so the hash holds on every checkout.

## Format

Two header rows. The first names the scales, the second the statistic under each one:

```
Words,Length,AROU,,,VAL,,,DOM,,,CNC,,,IMAG,,,FAM,,,AOA,,,SIZE,,,GEND,,
,,M,SD,N,M,SD,N,M,SD,N,M,SD,N,M,SD,N,M,SD,N,M,SD,N,M,SD,N,M,SD,N
```

`M` is the mean rating, `SD` its standard deviation and `N` the number of raters. The scales are
arousal (`AROU`), valence (`VAL`), dominance (`DOM`), concreteness (`CNC`), imageability (`IMAG`),
familiarity (`FAM`), age of acquisition (`AOA`), semantic size (`SIZE`) and gender association
(`GEND`).

**`GEND`** measures "the degree to which words are considered to be associated with male or female
behavior", on a 1–7 scale: **1 is feminine, 7 is masculine** (*queen* 1.31, *king* 6.81).

## License

The article is open access under the Creative Commons Attribution 4.0 International License, which
covers its supplementary material. See [`LICENSE.md`](LICENSE.md).
