"""
--------------
Parse a single `.lean` file and return structured metadata for every
`lemma`, `theorem`, or `def` header it contains.

gather_decls(fpath: str, module_name: str, mathlib_dir: str) -> list[dict]
    fpath        absolute path to the `.lean` file
    module_name  dotted Lean module prefix (e.g. Mathlib.Topology.Basic)
    mathlib_dir  root of the mathlib tree; used to make a relative path

Returns
    List of dicts with keys:
      • decl : fully-qualified Lean name
      • type : type signature (as plain text)
      • doc  : first-line doc-string or ""
      • path : path relative to mathlib root
"""

import re, pathlib, os

# regex for “lemma/theorem/def name : <type> := …”
DECL_RE  = re.compile(r'^(theorem|lemma|def)\s+([A-Za-z0-9_\.]+)\s*(:.*?)?:=', re.M | re.S)
# multiline doc-comment block     /-! … -/
DOCBLOCK = re.compile(r"/-!\s*(.*?)\s*-/", re.S)
# single-line doc-comment         --! …
DOCLINE  = re.compile(r"^\s*--!\s*(.*)")

def gather_decls(fpath: str, module_name: str, mathlib_dir: str) -> list[dict]:
    """Extract every declaration in one Lean file."""
    text = pathlib.Path(fpath).read_text(encoding="utf8")
    records, doc_map = [], {}

    # collect first-line of each multiline doc block
    for m in DOCBLOCK.finditer(text):
        doc = m.group(1).strip().split('\n')[0]
        tail = text[m.end():]
        n = re.search(r'\b(theorem|lemma|def)\s+([^ :\n]+)', tail)
        if n:
            doc_map[n.group(2)] = doc

    # collect single-line --! comments
    for line in text.splitlines():
        m = DOCLINE.match(line)
        if m:
            doc = m.group(1).strip()
            tail = line[m.end():]
            n = re.search(r'\b(theorem|lemma|def)\s+([^ :\n]+)', tail)
            if n:
                doc_map[n.group(2)] = doc

    # record each declaration header and its cached doc
    for m in DECL_RE.finditer(text):
        name  = m.group(2)
        type_ = (m.group(3) or '').lstrip(':').strip()
        records.append({
            "decl": f"{module_name}.{name}",
            "type": type_,
            "doc":  doc_map.get(name, ""),
            "path": os.path.relpath(fpath, mathlib_dir),
        })
    return records
