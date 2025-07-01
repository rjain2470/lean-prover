"""
scripts/extract_jsonl.py
------------------------
Walk the mathlib source tree, parse every `.lean` file with
`src.extract.gather_decls`, and write a JSONL dataset suitable for the
embedding step.

Each output line contains:
    {
        "id"   : sequential integer row id,
        "decl" : fully-qualified Lean name,
        "type" : type signature,
        "doc"  : first-line doc-string (may be empty),
        "path" : path relative to mathlib root,
        "text" : type + "\n" + doc   # the field consumed by the embedder
    }

CLI arguments
-------------
--root       Path to the mathlib folder to traverse (default: mathlib4/Mathlib)
--out        Destination JSONL file (default: datasets/type_doc.jsonl)
--subdir     Optional sub-folder filter; only files whose path contains this
             string are processed.

Usage
-----
    python scripts/extract_jsonl.py
    python scripts/extract_jsonl.py --subdir Algebra --out algebra.jsonl
"""


#!/usr/bin/env python
import argparse, json, logging, os, pathlib
from tqdm import tqdm
from src.extract import gather_decls # per-file extractor


def collect_lean_files(root: pathlib.Path, subdir_filter: str = "") -> list[tuple[str, str]]:
    """Return [(file_path, dotted_module_name), …] under *root*."""
    base_len = len(str(root)) + 1              # drop leading “root/”
    result: list[tuple[str, str]] = []

    for file in root.rglob("*.lean"):
        if subdir_filter and subdir_filter not in file.parts:
            continue
        rel_path = str(file)[base_len:].replace(os.sep, ".")  # path → dotted
        result.append((str(file), rel_path[:-5]))             # strip ".lean"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Lean declarations into JSONL.")
    parser.add_argument("--root", default="mathlib4/Mathlib",
                        help="Path to mathlib4/Mathlib")
    parser.add_argument("--out",  default="datasets/type_doc.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--subdir", default="",
                        help="Restrict traversal to this sub-folder")
    args = parser.parse_args()

    root_path = pathlib.Path(args.root).resolve()
    lean_files = collect_lean_files(root_path, args.subdir)
    logging.info("Found %d Lean files to parse.", len(lean_files))

    records: list[dict] = []
    for fpath, mod in tqdm(lean_files, unit="file"):
        records.extend(gather_decls(fpath, mod, str(root_path)))

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf8") as fh:
        for i, rec in enumerate(records):
            rec["id"] = i
            text = f'{rec["type"]}\n{rec["doc"]}' if rec["doc"] else rec["type"]
            fh.write(json.dumps({**rec, "text": text}, ensure_ascii=False) + "\n")

    logging.info("Wrote %d rows to %s", len(records), args.out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
