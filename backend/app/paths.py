"""Where the repository's files are, whatever the working directory.

Every path the code reads or writes by default hangs off ``REPO_ROOT``, found from this file, so a
script or a test runs the same from any directory.
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
BOARDS_DIR = DATA_DIR / "boards"
MEASUREMENT_FRAME_PATH = BOARDS_DIR / "measurement_frame.json"
PROMPT_TEMPLATES_DIR = DATA_DIR / "prompt_templates"
MORPHOLEX_PATH = DATA_DIR / "morpholex" / "morpholex_en.tsv"

ALEMBIC_INI = REPO_ROOT / "backend" / "alembic.ini"
LOGS_DIR = REPO_ROOT / "logs"
