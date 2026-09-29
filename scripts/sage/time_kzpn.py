#!/usr/bin/env python3
"""Time the ORIGINAL KZPN Sage code (antieau/kzpn) on chosen rings and weights.

Run it with Sage's Python:

    KZPN_DIR=/path/to/kzpn sage -python scripts/sage/time_kzpn.py --output kzpn_times.json \\
        [--repeats 3] [--timeout 3600] [CASE ...]

A CASE is ``p,n,i`` (the ring Z_p[z]/(z + p, z^n) = Z/p^n) or ``p,n,i:e_0,e_1,...`` (the
Eisenstein polynomial e_0 + e_1 z + ...). Without cases, the benchmark set of
``benchmark/vs_sage.jl`` is run.

Every run is one worker process of ``make_fixtures_f1.py`` (the same loader of the unmodified
KZPN sources, the same working precision N as the fixtures, notebook cell 2), started one at a
time so that runs do not compete for the machine. The time is the worker's ``sage_seconds``:
the wall clock of ``syntomic_matrices`` and ``syntomic_cohomology``, without starting Sage and
loading KZPN. Runs shorter than ``--repeat-below`` seconds are repeated ``--repeats`` times and
the minimum is kept. Cases are run in the given order; after a run that times out, the larger
weights of the same ring are skipped.
"""

import argparse
import datetime
import json
import os
import platform
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_fixtures_f1 as mf  # noqa: E402
import kzpn_upstream  # noqa: E402

# The benchmark set: the fixture cases on which Sage takes at least 0.3 s, and two families of
# growing weight.
DEFAULT_CASES = (
    ["3,4,8", "7,3,9", "2,3,8", "2,4,6", "3,3,9", "2,3,6", "3,3,7", "5,3,8", "3,3,5", "2,3,4"]
    + ["2,2,%d" % i for i in (5, 10, 15, 20, 25, 30)]
    + ["3,3,%d" % i for i in (11, 13, 15, 17)]
)


def parse_case(spec, timeout):
    head, _, tail = spec.partition(":")
    p, n, i = (int(x) for x in head.split(","))
    coeffs = [int(x) for x in tail.split(",")] if tail else [p, 1]
    case = mf.make_case(p, i, n, coeffs, "benchmark", timeout)
    return case


def notebook_N(texts, p, n, i):
    """The working precision N of the fixtures (notebook cell 2) from KZPN's precision.py."""
    ns = {}
    exec(compile(texts["precision.py"], "<KZPN/precision.py>", "exec"), ns)
    return mf.notebook_precision(ns["nygaard_exponent"], p, n, i)["N"]


def run(case, tmpdir):
    res = mf.run_one(case, tmpdir)
    if res["status"] != "ok":
        return {"status": res["status"], "wall": res["wall"]}
    rec = res["record"]
    if rec.get("error"):
        return {"status": "error", "error": rec["error"]}
    return {
        "status": "ok",
        "seconds": rec["sage_seconds"],
        "N": rec["N"],
        "h1": rec["h1"],
        "h2": rec["h2"],
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("cases", nargs="*", help="p,n,i or p,n,i:e_0,e_1,... (default: the benchmark set)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--timeout", type=float, default=3600.0, help="seconds per run")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--repeat-below", type=float, default=60.0)
    ap.add_argument("--allow-modified", action="store_true",
                    help="time a KZPN copy whose files differ from the upstream commit")
    args = ap.parse_args()

    import sage.version

    kzpn_dir, _ = mf.resolve_kzpn_dir()
    texts, sha = mf.read_kzpn(kzpn_dir)
    if sha != kzpn_upstream.UPSTREAM_SHA256 and not args.allow_modified:
        raise SystemExit(
            "%s is not the KZPN code of commit %s (SHA-256 differ); set KZPN_DIR to a clean "
            "checkout or pass --allow-modified" % (kzpn_dir, kzpn_upstream.UPSTREAM_COMMIT)
        )
    out = {
        "provenance": {
            "date": datetime.date.today().isoformat(),
            "sage_version": sage.version.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "kzpn_upstream_commit": kzpn_upstream.UPSTREAM_COMMIT,
            "kzpn_sha256": sha,
            "kzpn_matches_upstream": sha == kzpn_upstream.UPSTREAM_SHA256,
            "timing": "minimum of sage_seconds over the runs (syntomic_matrices + "
            "syntomic_cohomology, one fresh worker process per run, one run at a time)",
            "timeout_seconds": args.timeout,
        },
        "cases": [],
    }
    log = lambda msg: print(msg, file=sys.stderr, flush=True)
    timed_out = set()
    with tempfile.TemporaryDirectory(prefix="kzpn_time_") as tmpdir:
        for spec in args.cases or DEFAULT_CASES:
            case = parse_case(spec, args.timeout)
            ring = (case["p"], case["n"], tuple(case["E"]))
            entry = {"p": case["p"], "n": case["n"], "i": case["i"], "E": case["E"]}
            entry["N"] = notebook_N(texts, case["p"], case["n"], case["i"])
            if ring in timed_out:
                entry["status"] = "skipped"
                out["cases"].append(entry)
                log("[skipped] %s" % spec)
                continue
            runs = [run(case, tmpdir)]
            first = runs[0]
            if first["status"] == "ok" and first["seconds"] < args.repeat_below:
                runs += [run(case, tmpdir) for _ in range(args.repeats - 1)]
            ok = [r for r in runs if r["status"] == "ok"]
            if ok:
                assert ok[0]["N"] == entry["N"], "the worker used another precision"
                entry.update({k: ok[0][k] for k in ("h1", "h2")})
                entry["seconds"] = min(r["seconds"] for r in ok)
                entry["runs"] = [r["seconds"] for r in ok]
                entry["status"] = "ok"
                log("[ok]      %-24s N=%-5d %9.2f s" % (spec, entry["N"], entry["seconds"]))
            else:
                entry["status"] = first["status"]
                if first["status"] == "timeout":
                    timed_out.add(ring)
                    entry["timeout_seconds"] = args.timeout
                log("[%s] %s" % (first["status"], spec))
            out["cases"].append(entry)
            with open(args.output, "w") as fh:  # written after every case
                json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
