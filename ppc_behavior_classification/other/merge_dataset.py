#!/usr/bin/env python3
"""
Merge 5_cls and normal_3_3_12+SSB-record into a unified 7-class dataset.
The `other` class (label 0) is capped at 35% of each split's total samples.

Other-class priority order (fill quota in order, stop when full):
  1. 5_cls  normal_*             all available, sorted by name
  2. normal 0501_other_*         descending by trailing number
  3. normal other_active / other_rest / other_jump  ratio 4:4:1,
                                  each group descending by trailing number

Unified label schema:
  0  other       1  eat    2  drink
  3  convulsion  4  limp   5  sneeze  6  vomit

Usage:
  python merge_dataset.py              # copy files to merged_dataset/
  python merge_dataset.py --dry-run    # preview counts, no files copied
"""

import re
import shutil
import sys
from pathlib import Path

ROOT      = Path(r"D:\Desktop\combine")
FIVE_CLS  = ROOT / "hierarchical-cls" / "5_cls"
NORMAL_DS = ROOT / "hierarchical-cls" / "normal" / "normal_3_3_12+SSB-record"
OUTPUT    = ROOT / "merged_dataset"

OTHER_RATIO = 0.35   # target: other / total = 35 %

LABEL_NAMES = {
    0: "other", 1: "eat", 2: "drink",
    3: "convulsion", 4: "limp", 5: "sneeze", 6: "vomit",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def trailing_number(name: str) -> int:
    """Return the last integer found in name, used for descending sort."""
    nums = re.findall(r'\d+', name)
    return int(nums[-1]) if nums else -1


def proportional_split(total: int, ratios: list) -> list:
    """
    Distribute `total` into len(ratios) integer buckets proportionally.
    Remainder is spread across the first buckets.
    """
    s = sum(ratios)
    counts = [total * r // s for r in ratios]
    for i in range(total - sum(counts)):
        counts[i % len(counts)] += 1
    return counts


def infer_non_other_label(name: str, source: str):
    """
    Return the unified label for non-other classes only.
    Returns None for anything that belongs to the other class or is unrecognised.
    """
    n = name.lower()
    if source == "5cls":
        if n.startswith("convulsion"):  return 3   # 5cls 1 → unified 3
        if n.startswith("limp"):        return 4   # 5cls 2 → unified 4
        if n.startswith("sneeze"):      return 5   # 5cls 3 → unified 5
        if n.startswith("vomit"):       return 6   # 5cls 4 → unified 6
        # normal_* → other, handled separately
    elif source == "normal":
        if n.startswith("0501_eat"):    return 1   # fix: was 0 → now 1
        if n.startswith("eat"):         return 1
        if n.startswith("drink"):       return 2
        # 0501_other_* / other_* → other, handled separately
    return None


# ── Sample collection ─────────────────────────────────────────────────────────

def collect_non_other(split: str) -> list:
    """Return [(Path, label), ...] for every non-other sample in both sources."""
    samples = []
    for src_dir, tag in [(FIVE_CLS / split, "5cls"), (NORMAL_DS / split, "normal")]:
        if not src_dir.exists():
            continue
        for folder in sorted(src_dir.iterdir()):
            if not folder.is_dir():
                continue
            label = infer_non_other_label(folder.name, tag)
            if label is not None:
                samples.append((folder, label))
    return samples


def collect_other(split: str, target: int) -> tuple:
    """
    Collect up to `target` other-class samples using the priority rules.
    Returns ([(Path, 0), ...], source_counts_dict).
    """
    result = []
    source_counts = {k: 0 for k in
                     ["5cls_normal", "0501_other", "other_active", "other_rest", "other_jump"]}

    def take(folders: list, n: int, key: str) -> None:
        chosen = folders[:n]
        result.extend((f, 0) for f in chosen)
        source_counts[key] = len(chosen)

    remaining = target

    # ── Priority 1: 5_cls normal_* ──────────────────────────────────────────
    five_dir = FIVE_CLS / split
    if five_dir.exists() and remaining > 0:
        p1 = sorted(
            [f for f in five_dir.iterdir()
             if f.is_dir() and f.name.lower().startswith("normal")],
            key=lambda f: f.name,
        )
        n = min(len(p1), remaining)
        take(p1, n, "5cls_normal")
        remaining -= n

    # ── Priority 2: 0501_other_* (descending by number) ─────────────────────
    norm_dir = NORMAL_DS / split
    if norm_dir.exists() and remaining > 0:
        p2 = sorted(
            [f for f in norm_dir.iterdir()
             if f.is_dir() and f.name.lower().startswith("0501_other")],
            key=lambda f: trailing_number(f.name),
            reverse=True,
        )
        n = min(len(p2), remaining)
        take(p2, n, "0501_other")
        remaining -= n

    # ── Priority 3: other_active / other_rest / other_jump 4:4:1 (desc) ────
    if norm_dir.exists() and remaining > 0:
        def get_desc(prefix: str) -> list:
            return sorted(
                [f for f in norm_dir.iterdir()
                 if f.is_dir() and f.name.lower().startswith(prefix)],
                key=lambda f: trailing_number(f.name),
                reverse=True,
            )

        active_all = get_desc("other_active")
        rest_all   = get_desc("other_rest")
        jump_all   = get_desc("other_jump")

        n_a, n_r, n_j = proportional_split(remaining, [4, 4, 1])
        n_a = min(n_a, len(active_all))
        n_r = min(n_r, len(rest_all))
        n_j = min(n_j, len(jump_all))

        take(active_all, n_a, "other_active")
        take(rest_all,   n_r, "other_rest")
        take(jump_all,   n_j, "other_jump")

    return result, source_counts


# ── File copying ──────────────────────────────────────────────────────────────

def copy_sample(src: Path, dst: Path, label: int) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if not f.is_file():
            continue
        if f.name == "label.txt":
            (dst / "label.txt").write_text(str(label), encoding="utf-8")
        else:
            shutil.copy2(f, dst / f.name)


# ── Per-split processing ──────────────────────────────────────────────────────

def process_split(split: str, dry_run: bool) -> dict:
    out_dir = OUTPUT / split

    non_other = collect_non_other(split)

    # other / total = OTHER_RATIO  =>  other = non_other * R / (1 - R)
    target_other = round(len(non_other) * OTHER_RATIO / (1 - OTHER_RATIO))
    other_samples, src_counts = collect_other(split, target_other)
    actual_other = len(other_samples)

    all_samples = non_other + other_samples
    total = len(all_samples)

    print(f"  Non-other   : {len(non_other):6d}")
    print(f"  Other target: {target_other:6d}  "
          f"(= {OTHER_RATIO:.0%} × {total})")
    print(f"  Other actual: {actual_other:6d}  "
          f"({actual_other / total:.1%} of total)")
    print(f"  Total       : {total:6d}")

    print(f"\n  Other source breakdown:")
    for key, count in src_counts.items():
        bar = "#" * min(count * 20 // max(src_counts.values(), default=1), 20)
        print(f"    {key:<14}: {count:5d}  {bar}")

    # ── copy / count ──────────────────────────────────────────────────────────
    stats = {i: 0 for i in range(7)}
    seen: dict = {}   # dst_name → collision count

    for src_path, label in all_samples:
        dst_name = src_path.name
        if dst_name in seen:
            seen[dst_name] += 1
            dst_name = f"{dst_name}_dup{seen[dst_name]}"
        else:
            seen[dst_name] = 0

        if not dry_run:
            copy_sample(src_path, out_dir / dst_name, label)
        stats[label] += 1

    # ── label distribution table ──────────────────────────────────────────────
    max_count = max(stats.values(), default=1)
    print(f"\n  Label distribution  (total = {total})")
    print(f"  {'lbl':>3}  {'class':>12}  {'count':>7}  {'%':>5}  bar")
    print(f"  {'─' * 58}")
    for lbl in range(7):
        count = stats[lbl]
        pct   = count / total * 100
        bar   = "#" * round(count * 36 / max_count)
        print(f"  {lbl:>3}  {LABEL_NAMES[lbl]:>12}  {count:7d}  {pct:4.1f}%  {bar}")

    return stats


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    dry_run = "--dry-run" in sys.argv
    print(f"{'═' * 62}")
    print(f" merge_dataset.py  —  other capped at {OTHER_RATIO:.0%}")
    print(f"{'─' * 62}")
    print(f" Mode   : {'DRY RUN — no files copied' if dry_run else 'LIVE — copying files'}")
    print(f" Source1: {FIVE_CLS}")
    print(f" Source2: {NORMAL_DS}")
    print(f" Output : {OUTPUT}")
    print(f"{'═' * 62}")

    if not dry_run and OUTPUT.exists():
        ans = input("\n[WARN] Output directory already exists. Continue? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return

    all_stats: dict = {}
    for split in ("train", "val"):
        print(f"\n{'─' * 62}")
        print(f" Split: {split}")
        print(f"{'─' * 62}")
        all_stats[split] = process_split(split, dry_run)

    print(f"\n{'═' * 62}")
    print(" SUMMARY")
    print(f"{'─' * 62}")
    grand = 0
    for split, stats in all_stats.items():
        n = sum(stats.values())
        grand += n
        other_pct = stats[0] / n * 100 if n else 0
        print(f"  {split:6s}: {n:6d} samples  (other {other_pct:.1f}%)")
    print(f"  {'total':6s}: {grand:6d} samples")
    print(f"{'═' * 62}")

    if dry_run:
        print("\nRe-run without --dry-run to copy the files.")


if __name__ == "__main__":
    main()
