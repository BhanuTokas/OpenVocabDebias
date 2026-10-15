"""
results_io.py — Append-only CSV logging for summary.csv files.
"""

from __future__ import annotations

import csv
import os
import time


def append_result(path: str, row: dict):
    """Append `row` to the CSV at `path`, keeping columns aligned.

    If the file already exists with a different header (e.g. a column was
    added to the row schema), the file is rewritten with the union of the old
    and new columns (old order first, missing cells left blank) instead of
    appending a row that no longer lines up with the header.  If the existing
    file is already misaligned (rows longer than its header), it is moved
    aside to `<name>.bak-<timestamp>.csv` and a fresh file is started.
    """
    fieldnames = list(row.keys())

    if os.path.isfile(path) and os.path.getsize(path) > 0:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames or [])
            header_covers_row = set(fieldnames) <= set(existing_fields)
            existing_rows = [] if header_covers_row else list(reader)

        if header_covers_row:
            with open(path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=existing_fields, restval="").writerow(row)
            return

        if any(None in r for r in existing_rows):
            root, ext = os.path.splitext(path)
            backup = f"{root}.bak-{time.strftime('%Y%m%d-%H%M%S')}{ext}"
            os.replace(path, backup)
            print(
                f"  [append_result] {path} has rows longer than its header; "
                f"moved it to {backup} and starting a fresh file"
            )
        else:
            merged = existing_fields + [
                k for k in fieldnames if k not in existing_fields
            ]
            print(
                f"  [append_result] header of {path} differs from row schema; "
                f"rewriting with {len(merged)} merged columns"
            )
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=merged, restval="")
                writer.writeheader()
                writer.writerows(existing_rows)
                writer.writerow(row)
            return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
