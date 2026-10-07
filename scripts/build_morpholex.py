"""Builds data/morpholex/morpholex_en.tsv, the word segmentations the clue validator reads.

The source is MorphoLex-en (https://github.com/hugomailhot/MorphoLex-en), distributed as one Excel
workbook, MorphoLEX_en.xlsx, with one sheet per morphological structure. This keeps two of its
columns, Word and MorphoLexSegm, from every sheet that has them, lowercases the words and drops the
rows whose Word is not a word (a handful hold a segmentation or a number). The output is sorted by
word, with a header line.

The workbook is read with the standard library (an .xlsx is a zip of XML files), so the script
needs no dependency. The data is licensed CC BY-NC-SA 4.0, and so is the output: see
data/morpholex/README.md.

Run from the REPO ROOT:
    python scripts/build_morpholex.py path/to/MorphoLEX_en.xlsx
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_WORD = re.compile(r"[a-z][a-z'-]*")
_DEFAULT_OUT = Path("data/morpholex/morpholex_en.tsv")


def read_segmentations(xlsx: Path) -> tuple[dict[str, str], int]:
    """
    Reads Word -> MorphoLexSegm from every sheet of the workbook that has both columns.

    :param xlsx: Path to MorphoLEX_en.xlsx.
    :return: The segmentations by lowercase word, and the number of rows dropped because their Word
        is not a word.
    """
    book = zipfile.ZipFile(xlsx)
    shared = [
        "".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t"))
        for si in ET.fromstring(book.read("xl/sharedStrings.xml")).findall("m:si", _NS)
    ]
    sheets = ET.fromstring(book.read("xl/workbook.xml")).find("m:sheets", _NS)
    segmentations: dict[str, str] = {}
    dropped = 0
    for index in range(1, len(sheets) + 1):
        path = f"xl/worksheets/sheet{index}.xml"
        if path not in book.namelist():
            continue
        header = None
        for row in ET.fromstring(book.read(path)).find("m:sheetData", _NS):
            cells = {}
            for cell in row:
                value = cell.find("m:v", _NS)
                if value is None:
                    continue
                column = re.match(r"[A-Z]+", cell.get("r")).group()
                cells[column] = shared[int(value.text)] if cell.get("t") == "s" else value.text
            if header is None:
                header = {name: column for column, name in cells.items()}
                if "Word" not in header or "MorphoLexSegm" not in header:
                    break
                continue
            word = (cells.get(header["Word"]) or "").strip().lower()
            segmentation = (cells.get(header["MorphoLexSegm"]) or "").strip()
            if not word or not segmentation:
                continue
            if not _WORD.fullmatch(word):
                dropped += 1
                continue
            segmentations.setdefault(word, segmentation)
    return segmentations, dropped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("xlsx", type=Path, help="path to MorphoLEX_en.xlsx")
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT,
                        help=f"output TSV (default: {_DEFAULT_OUT})")
    args = parser.parse_args(argv)

    segmentations, dropped = read_segmentations(args.xlsx)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("word\tsegmentation\n")
        for word in sorted(segmentations):
            f.write(f"{word}\t{segmentations[word]}\n")
    print(f"{len(segmentations)} words written to {args.out} ({dropped} rows dropped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
