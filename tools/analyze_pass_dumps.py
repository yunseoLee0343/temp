#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TOKENS = {
    "latency_attr": r"tt\.latency|latency\s*=",
    "loop_stage_attr": r"tt\.loop_stage|loop_stage",
    "loop_cluster_attr": r"tt\.loop_cluster|loop_cluster",
    "scheduled_max_stage": r"scheduled.*stage|tt\.scheduled",
    "async_copy_g2l": r"async_copy_global_to_local|AsyncCopyGlobalToLocal",
    "async_commit": r"async_commit_group|AsyncCommitGroup",
    "async_wait": r"async_wait|AsyncWait",
    "local_alloc": r"ttg\.local_alloc",
    "local_load": r"ttg\.local_load",
    "memdesc": r"!ttg\.memdesc<",
    "software_lowerloops_marker": r"SoftwarePipeliner internal IR Dump After: LowerLoops",
    "software_expandloops_marker": r"SoftwarePipeliner internal IR Dump After: ExpandLoops",
}


def summarize(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    counts = {name: len(re.findall(pattern, text, flags=re.IGNORECASE))
              for name, pattern in TOKENS.items()}
    evidence = []
    for i, line in enumerate(text.splitlines(), start=1):
        if any(re.search(pattern, line, flags=re.IGNORECASE)
               for pattern in TOKENS.values()):
            evidence.append({"line": i, "text": line.strip()})
    return {
        "file": str(path),
        "bytes": path.stat().st_size,
        "counts": counts,
        "evidence": evidence,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--glob", default="mini_chunk_fla_forced.stage-*.mlir.txt")
    args = ap.parse_args()

    rows = []
    for path in sorted(args.dump_dir.glob(args.glob)):
        m = re.search(r"stage-(\d+)", path.name)
        stage = int(m.group(1)) if m else None
        row = summarize(path)
        row["stage"] = stage
        rows.append(row)

    if not rows:
        raise SystemExit(f"no pass dumps found under {args.dump_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    txt = args.out.with_suffix(".txt")
    lines = []
    for row in rows:
        lines.append(f"stage={row['stage']} bytes={row['bytes']}")
        lines.append(f"  counts={row['counts']}")
        for ev in row["evidence"][:80]:
            lines.append(f"  L{ev['line']}: {ev['text']}")
        lines.append("")
    txt.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {args.out}")
    print(f"Wrote: {txt}")


if __name__ == "__main__":
    main()
