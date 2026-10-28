# Occupational status (SIOPS-08)

The status covariate of an occupation word comes from Treiman's Standard International Occupational
Prestige Scale, in its ISCO-08 version (SIOPS-08), decision D-03. The scale is not stored here:
each occupation word of the lexicon carries the ISCO-08 code assigned to it by hand and the SIOPS-08
score of that code, with this directory as the source. Words that are not occupations carry no
score and are flagged as such.

## Source

Harry Ganzeboom's conversion tools for ISCO-08, <http://www.harryganzeboom.nl/isco08/index.htm>,
file `isco08.zip` (dated 2012-01-06), downloaded on 2026-10-28:

| File | Contents | SHA-256 |
|---|---|---|
| `isco08.zip` | the SPSS syntax modules | `5f1805cc1837bee0afdb27efc2499ee72b2246a61e5e453784bd6a036814f69b` |
| `isqotrei08.sps` (in the zip) | ISCO-08 code → SIOPS-08 score, 590 codes | `65f93cf9a082904deea85cd14245931ab2f402d42c7f4d1c093c2ac318ed683a` |
| `isqolab.sps` (in the zip) | English titles of the ISCO-08 codes | `74445d206118253476cc727709cd13375addb8fba2607521ad515fa9f9b59ffd` |

Cite as:

> Ganzeboom, H. B. G., & Treiman, D. J. International Stratification and Mobility File: Conversion
> Tools. <http://www.harryganzeboom.nl/ismf/index.htm>

> Treiman, D. J. (1977). *Occupational Prestige in Comparative Perspective*. New York: Academic Press.

## License

The tools are offered for public use, without an explicit license, so the files are not copied
into the repository: only the score of each occupation word, with its code, travels in the lexicon.
