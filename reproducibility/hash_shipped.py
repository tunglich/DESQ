"""Print SHA-256 fingerprints of the shipped data files that back
``EXPECTED_OUTPUT.md``. Compare against the pinned prefixes to detect any
post-hoc tampering with prices or features.

Usage:
    python reproducibility/hash_shipped.py                 # default set
    python reproducibility/hash_shipped.py --all-features  # every features/*.csv
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SET = [
    'tw50_top50.csv',
    'prices/2330.csv',
    'features/fundamental_2330.csv',
    'features/trade_2330.csv',
    'features/tech_trend_2330.csv',
    'features/moment_2330.csv',
    'features/macro_2330.csv',
]


def _sha256(fp: Path) -> str:
    h = hashlib.sha256()
    with fp.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _count_lines(fp: Path) -> int:
    with fp.open('rb') as fh:
        return sum(1 for _ in fh)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--all-features', action='store_true',
                   help='hash every prices/*.csv and features/*.csv')
    p.add_argument('--full', action='store_true', help='print full 64-hex hash')
    args = p.parse_args()

    if args.all_features:
        rels = sorted({p.relative_to(REPO_ROOT).as_posix() for p in
                       (REPO_ROOT / 'prices').glob('*.csv')}) + \
               sorted({p.relative_to(REPO_ROOT).as_posix() for p in
                       (REPO_ROOT / 'features').glob('*.csv')})
    else:
        rels = DEFAULT_SET

    width_prefix = 64 if args.full else 16
    print(f'{"sha256":<{width_prefix}}  {"lines":>7}  file')
    print(f'{"-" * width_prefix}  {"-" * 7}  {"-" * 32}')
    for rel in rels:
        fp = REPO_ROOT / rel
        if not fp.exists():
            print(f'{"MISSING":<{width_prefix}}  {"":>7}  {rel}')
            continue
        h = _sha256(fp)
        print(f'{(h if args.full else h[:16]):<{width_prefix}}  '
              f'{_count_lines(fp):>7}  {rel}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
