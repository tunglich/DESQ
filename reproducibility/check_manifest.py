"""Cross-platform manifest verifier (mirror of ``sha256sum -c``).

Reads ``reproducibility/MANIFEST.sha256`` (standard format:
``<64-hex>  <relative/path>``), recomputes SHA-256 for each listed file, and
reports any mismatch or missing file. Used both by CI (linux) and Windows
users for whom ``sha256sum -c`` is not available.

Exit codes: 0 = all pass, 1 = mismatch/missing, 2 = manifest file missing.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(fp: Path) -> str:
    h = hashlib.sha256()
    with fp.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--manifest',
                   default='reproducibility/MANIFEST.sha256',
                   help='path to manifest (relative to repo root)')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    mf = REPO_ROOT / args.manifest
    if not mf.exists():
        print(f'::error::manifest missing: {mf}')
        return 2

    total = 0
    fails: list[tuple[str, str]] = []
    with mf.open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                fails.append((line, 'malformed line'))
                continue
            expected, rel = parts[0].lower(), parts[1].strip()
            fp = REPO_ROOT / rel
            total += 1
            if not fp.exists():
                fails.append((rel, 'MISSING'))
                if args.verbose:
                    print(f'  MISSING  {rel}')
                continue
            actual = _sha256(fp)
            if actual != expected:
                fails.append((rel, f'sha256 mismatch (expected {expected[:16]}, got {actual[:16]})'))
                if args.verbose:
                    print(f'  FAIL     {rel}')
            elif args.verbose:
                print(f'  OK       {rel}')

    if fails:
        print(f'\n[FAIL] {len(fails)}/{total} files did not match:')
        for rel, reason in fails[:20]:
            print(f'  {rel}: {reason}')
        if len(fails) > 20:
            print(f'  ... and {len(fails) - 20} more')
        return 1
    print(f'MANIFEST OK: {total} files verified against {args.manifest}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
