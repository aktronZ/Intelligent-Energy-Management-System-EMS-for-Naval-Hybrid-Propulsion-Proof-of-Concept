"""Run every acceptance gate.

Usage: uv run python -m acceptance.run
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The console is cp1252 on Windows but the thesis prose is UTF-8. Without this,
# printing a review clause containing a character outside cp1252 (for example
# the <= of "offline optimum <= MPC") raises UnicodeEncodeError and the gate
# reports FAILED. Reconfigure stdout so the review report can be printed.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from acceptance import (  # noqa: E402
    check_claims,
    check_crossrefs,
    check_figures,
    check_optimality,
    check_provenance,
    check_prototype_tense,
)

GATES = [
    ("provenance", check_provenance.check),
    ("crossrefs", lambda: check_crossrefs.check()),
    ("claims", lambda: check_claims.check()),
    ("optimality (G1, G2, G5, G6)", check_optimality.check),
    ("figures (G7)", check_figures.check),
    ("prototype tense (Chapter 8)", check_prototype_tense.check),
]


def main() -> int:
    print("=" * 68)
    print("ACCEPTANCE GATES")
    print("=" * 68 + "\n")
    failed: list[str] = []
    for name, gate in GATES:
        print("-" * 68)
        print(f"GATE  {name}")
        print("-" * 68)
        try:
            rc = gate()
        except Exception as exc:  # a crash is a failure, not a skip
            print(f"ERROR in {name}: {exc}")
            rc = 1
        print()
        if rc != 0:
            failed.append(name)

    print("=" * 68)
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("ALL GATES PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
