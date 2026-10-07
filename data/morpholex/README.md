# MorphoLex-en segmentations

`morpholex_en.tsv` gives the morphological segmentation of 68,615 English words. The clue validator
(`backend/app/core/clue_validator.py`) reads it to decide whether a clue is a form of a visible word
or a part of a visible compound (§8.1.5–6 of the Duet rules).

Each line is a word and its segmentation in MorphoLex notation: `{...}` is a root group, `(x)` a
root, `<x<` a prefix and `>x>` a suffix. Affixes inside the braces are fused with the root:

| word        | segmentation          |
|-------------|-----------------------|
| earthquake  | `{(earth)}{(quake)}`  |
| earthy      | `{(earth)}>y>`        |
| unstable    | `<un<{(stable)}`      |
| transport   | `{<trans<(port)}`     |

## Source

MorphoLex-en, <https://github.com/hugomailhot/MorphoLex-en>, file `MorphoLEX_en.xlsx`, by Claudia
H. Sánchez-Gutiérrez, Hugo Mailhot, S. Hélène Deacon and Maximiliano A. Wilson. The authors ask to
cite it as:

> Sánchez-Gutiérrez, C.H., Mailhot, H., Deacon, S.H. et al. Behav Res (2017).
> <https://doi.org/10.3758/s13428-017-0981-8>

The article is *MorphoLex: A derivational morphological database for 70,000 English words*,
Behavior Research Methods.

## Changes from the original

`morpholex_en.tsv` is an adaptation of `MorphoLEX_en.xlsx`:

* Only two columns are kept, `Word` and `MorphoLexSegm`, from every sheet that has them. The
  frequency and family-size variables are left out.
* Words are lowercased, and the 9 rows whose `Word` is not a word (a segmentation or a number) are
  dropped.
* The rows are sorted by word and saved as tab-separated text.

The segmentations themselves are unchanged. `scripts/build_morpholex.py` rebuilds the file from the
workbook:

```bash
python scripts/build_morpholex.py path/to/MorphoLEX_en.xlsx
```

## License

MorphoLex-en is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International License (CC BY-NC-SA 4.0), and so is `morpholex_en.tsv`, as the license requires for
adapted material. See [`LICENSE.md`](LICENSE.md).

This applies only to the files in this directory. The rest of the repository is under the MIT
License (see the root `LICENSE`). The NonCommercial term means the file cannot be used for
commercial purposes: anyone reusing the platform commercially has to remove it or replace it with
data under a license that allows that use.
