# Small World of Words (SWOW-EN18)

The generator computes the coverability κ of a pair of cards from SWOW-EN18, the English word
association norms of the Small World of Words project (decision D-02): `sim` is the cosine between
the association profiles of two words. **The data are not in the repository**: their license does
not allow redistributing them.

## Getting the file

1. Download **SWOW-EN18** (version 1.0, released 2018-10-18) from
   <https://smallworldofwords.org/en/project/research>. The download form asks for a name, an email
   and what the data will be used for.
2. Put `SWOW-EN18.zip` in this directory, unchanged. Git ignores everything here except this README
   and the `.gitignore`.
3. The generator checks the file before using it:
   * **SHA-256:** `69ee1f4f19c657bb2f6e1bc27abd2becbcd01d83d1fdade94a0455b4e66eb093`
   * **Size:** 80,349,486 bytes

The zip holds the raw and preprocessed responses, cue and response statistics, and the associative
strengths (`strength.SWOW-EN.R1.20180827.csv`, `strength.SWOW-EN.R123.20180827.csv`: one row per
cue and response, tab-separated). The generator reads the R123 strengths from inside the zip.

## Citation

> De Deyne, S., Navarro, D. J., Perfors, A., Brysbaert, M., & Storms, G. (2019). The "Small World of
> Words" English word association norms for over 12,000 cue words. *Behavior Research Methods*, 51,
> 987–1006. <https://doi.org/10.3758/s13428-018-1115-7>

Report the dataset as SWOW-EN18, version 1.0, released 2018-10-18.

## License

Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported
(<https://creativecommons.org/licenses/by-nc-nd/3.0/>). The project's terms of access: "Free for
non-commercial scientific research; download requires name and email; no redistribution without
permission."

Whether the κ values computed from it may travel in the published boards is still open: see the
pending note on GEN-05 in the project plan.
