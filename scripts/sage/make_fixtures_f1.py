#!/usr/bin/env python3
"""Generate f = 1 fixtures from the ORIGINAL KZPN Sage code (antieau/kzpn).

Run it with Sage's Python (never with ``sage x.sage``, which writes files next
to the source):

    sage -python scripts/sage/make_fixtures_f1.py --date 2026-09-25 [--jobs 4]

The output is ``test/fixtures/sage_f1.json`` (override with ``--output``).

How the KZPN code is loaded
---------------------------
* The KZPN directory is ``$KZPN_DIR``, defaulting to ``../Background/KZPN``
  relative to the repository root (the Syntomic.jl folder). The fixtures are
  generated from a clean checkout of https://github.com/antieau/kzpn at commit
  ed381b4bf8df9522a12829ee0cbc9a2374949231:

      git clone https://github.com/antieau/kzpn /path/to/kzpn
      git -C /path/to/kzpn checkout ed381b4bf8df
      KZPN_DIR=/path/to/kzpn sage -python scripts/sage/make_fixtures_f1.py --date YYYY-MM-DD

  The provenance records the upstream commit, the SHA-256 of the files at that
  commit and of the files actually read, and whether they agree
  (``kzpn_upstream.py``).
* ``precision.py`` and ``matrix_functions.py`` are read as text and exec'd
  into in-memory modules registered in ``sys.modules`` (so the
  ``from precision import *`` lines of the .sage file resolve without touching
  the file system). Bytecode writing is disabled.
* ``prismatic_envelope.sage`` is read as text, the single anchored
  instrumentation patch below is applied in memory, the text is preparsed with
  ``sage.repl.preparse.preparse_file`` and exec'd into a namespace ``G``;
  ``sage.repl.user_globals.set_globals(G)`` makes ``inject_variables`` write
  into ``G``. Nothing is ever written next to the KZPN sources.

Patch set (instrumentation only, no mathematical change)
-------------------------------------------------------
``syntomic_matrices`` only returns (syn0, syn1, nablaN, nablaP). One line is
inserted immediately before its Step 7 line (the anchor, which ends before any
trailing comment of that line, must occur exactly once, at the start of a line)
that hands can0, phi0, can1, OKtoOKmodpi0, OKtoOKmodpi1 and nablaP_OK to a
capture hook. The returned matrices give the rest.

Global variables
----------------
The Sage code reads the GLOBALS ``n`` and ``i`` inside prismatic_envelope_g
(L269: num_g) and ``i`` inside prismatic_envelope_f (L53-56), and ``W``, ``A``,
``z`` everywhere. Each case runs in a fresh worker process in which the
globals p, i, n (= Sage k), W, A, z, E, Fprec are set before the call.

Precision policy
----------------
Notebook KThy-Computation.ipynb cell 2 ("total_precision"), with the float
``math.floor(math.log(Fprec-1, p))`` replaced by an exact integer floor-log:

    N = T + ilog_p(L-1) + sum_{q=p}^{i-1} n*v_p(q!) + n*floor(i(i-1)/2) + T + 1,

T = nygaard_exponent(p, i, n) from precision.py (target precision, Lemma 7.2).

Each case is run in its own subprocess with a wall-clock timeout (120 s; 160 s
for the optional case). Timed-out cases are listed under "skipped"; cases on
which Sage raises are recorded with an "error" field.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
import types
from concurrent.futures import ThreadPoolExecutor

sys.dont_write_bytecode = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # kzpn_upstream.py
from kzpn_upstream import KZPN_FILES, kzpn_provenance, warn_if_not_upstream  # noqa: E402

SCRIPT_PATH = os.path.abspath(__file__)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(SCRIPT_PATH), "..", ".."))
DEFAULT_KZPN_REL = os.path.join("..", "Background", "KZPN")
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "test", "fixtures", "sage_f1.json")

FORMAT_ID = "syntomic-fixture-v1"
PATCH_SET_ID = "kzpn-f1-instrument-v1"

# The Step 7 statement of syntomic_matrices (prismatic_envelope.sage L733),
# without the line's end: copies of the file may carry a trailing comment there.
# The a_/b_ copies in syntomic_matrices_v1 carry prefixes, so this statement
# occurs exactly once, at the start of a line (both asserted at load time).
ANCHOR_STEP7 = (
    "    OKtoOKmodpi0,OKtoOKmodpi1,nablaP_OK,nablaP="
    "square_complete_bottom(OKtoOKmodpi0,OKtoOKmodpi1,nablaP_OK)"
)
CAPTURE_LINE = (
    "    _KZPN_CAPTURE_HOOK(can0=can0, phi0=phi0, can1=can1, red=OKtoOKmodpi0, "
    "redN=OKtoOKmodpi1, nablaOK=nablaP_OK)\n"
)
PATCHES = [
    {
        "id": "capture-before-step7",
        "kind": "instrumentation",
        "file": "prismatic_envelope.sage",
        "function": "syntomic_matrices",
        "description": (
            "Insert one call to an external capture hook immediately before the Step 7 "
            "square_complete_bottom line (L733), handing over can0, phi0, can1, "
            "OKtoOKmodpi0 (red), OKtoOKmodpi1 (redN) and nablaP_OK. The hook stores copies; "
            "no value used by the computation is changed."
        ),
        "anchor": ANCHOR_STEP7,
        "insert_before_anchor": CAPTURE_LINE,
    }
]

SIGN_CONVENTION = (
    "nablaOK is expressed in the generators g_u of the Sage implementation: for odd p they are the "
    "negatives of the generators of [AKN, Construction 3.28] and nablaOK is the matrix of "
    "[AKN, Section 7.1, Step 5]; for p = 2 they differ by units, so only the cohomology is compared."
)

PRECISION_POLICY = (
    "KThy-Computation.ipynb cell 2 total_precision with exact integer floor-log: "
    "N = T + ilog_p(L-1) + sum_{q=p}^{i-1} n*v_p(q!) + n*floor(i*(i-1)/2) + T + 1, "
    "T = nygaard_exponent(p,i,n) (precision.py). W = Zp(p, N, type='capped-abs'), "
    "A = PowerSeriesRing(W, L) with L = n*i."
)

INDEX_CONVENTIONS = {
    "can0": (
        "Sage can0 (L628-646). rows: index m-1 <-> F-weight m = 1..L-1 of the reduced Prism_R basis "
        "z^(m mod n) * prod_u f_u^(digit_u(floor(m/n))); cols: index j-1 <-> F-weight j = 1..L-1 of the "
        "N^{>=i} basis element E^(i-floor(j/n)) * z^(j mod n) * prod_u f_u^(digit_u(floor(j/n))); "
        "column j = coordinates of NF_R(that element)."
    ),
    "phi0": (
        "Sage phi0 (L653-671). rows: index m-1 <-> F-weight m = 1..L-1 (Prism_R basis); cols: index j-1 "
        "<-> F-weight j = 1..L-1 of the N^{>=i} basis; column j = NF_R(z^(p*(j mod n)) * "
        "phi_divided(prod_u f_u^(digit_u(floor(j/n)))))."
    ),
    "can1": (
        "Sage can1 (L673-693). rows: index m <-> F-weight m = 0..L-2 (Prism_R basis); cols: index j <-> "
        "F-weight j = 0..L-2 of the N^{>=i-1} basis element E^(i-1-floor(j/n)) * z^(j mod n) * "
        "prod_u f_u^(digit_u(floor(j/n)))."
    ),
    "red": (
        "Sage OKtoOKmodpi0 (L706-717). rows: index m-1 <-> F-weight m = 1..L-1 (Prism_R basis); cols: "
        "index j-1 <-> Prism_OK basis element z^j, j = 1..L-1; column j = NF_R(z^j)."
    ),
    "redN": (
        "Sage OKtoOKmodpi1 (L719-730). rows: index m <-> F-weight m = 0..L-2 (Prism_R basis); cols: "
        "index j <-> z^j, j = 0..L-2; column j = NF_R(z^j)."
    ),
    "nablaOK": (
        "Sage nablaP_matrix_OK output (L587-600). rows: index m = 0..L-2 <-> coefficient of z^m in the "
        "g0-coefficient (Sage sign convention g0 = z0 - z1); cols: index j-1 <-> z^j, j = 1..L-1 "
        "(F-weight j); column j = g0-coefficient of NF_OK(-(z - g0)^j * v^i)."
    ),
    "nablaR": (
        "Sage nablaP after square_complete_bottom (Step 7): redN * nablaOK * red^-1. rows as redN rows "
        "(index m <-> F-weight m = 0..L-2); cols as red rows (index j-1 <-> F-weight j = 1..L-1)."
    ),
    "nablaN": (
        "Sage nablaN from square_complete_top (Step 8): can1^-1 * nablaR * can0. rows as can1 cols "
        "(index m <-> N^{>=i-1} basis F-weight m = 0..L-2); cols as can0 cols (index j-1 <-> N^{>=i} "
        "basis F-weight j = 1..L-1)."
    ),
    "canmphi": (
        "Sage syn0 = can0 - phi0 (Step 9, L736). rows/cols as can0 (index = F-weight - 1, weights 1..L-1)."
    ),
    "canmphiN": (
        "Sage syn1 from square_complete_right (Step 10): nablaR * canmphi * nablaN^-1. rows as nablaR rows "
        "(index m <-> F-weight m = 0..L-2); cols as nablaN rows (index m <-> N^{>=i-1} basis F-weight "
        "m = 0..L-2)."
    ),
}

DIFFERENTIALS = (
    "syntomic_cohomology (L1028-1069): d0 = block_matrix([[canmphi],[nablaN]]) (2(L-1) x (L-1)), "
    "d1 = block_matrix([[nablaR, -canmphiN]]) ((L-1) x 2(L-1)); H^2 = torsion of the elementary divisors "
    "of d1, H^1 = torsion of the elementary divisors of d0 (both over W = Zp capped-abs); exponents are "
    "p-adic valuations of the elementary divisors different from 0 and 1."
)

# ---------------------------------------------------------------------------
# Case list
# ---------------------------------------------------------------------------

UNRAMIFIED_PIN = [
    (7, 3, 2), (2, 2, 2), (3, 2, 2), (2, 3, 2), (3, 4, 2), (2, 2, 3), (5, 2, 3), (2, 4, 3),
    (3, 5, 3), (2, 6, 3), (5, 6, 2), (3, 7, 3), (2, 8, 3), (3, 9, 3), (5, 8, 3), (2, 6, 4),
    (7, 9, 3), (3, 1, 2), (2, 1, 3), (5, 2, 2), (7, 2, 2), (11, 2, 3),
]
# (p, E coefficients low->high, [(i, n), ...])
RAMIFIED = [
    (2, [2, 2, 1], [(1, 3), (2, 2), (2, 3), (3, 2), (2, 4)]),
    (3, [3, 0, 1], [(2, 2), (2, 3), (3, 2)]),
    (3, [3, 3, 0, 1], [(2, 2), (2, 4)]),
    (5, [5, 5, 1], [(2, 3), (3, 2)]),
]
N1_PIN = [(2, 2, 1), (3, 3, 1)]
OPTIONAL_PIN = [(3, 8, 4)]
# Same ring R = Z_2[z]/(E, z^3) for both E (they agree mod 4 = 2^ceil(3/2)).
E_PAIR = {"p": 2, "n": 3, "is": [1, 2], "E_a": [2, 0, 1], "E_b": [2, 4, 1]}

DEFAULT_TIMEOUT = 120
OPTIONAL_TIMEOUT = 160


def e_label(coeffs):
    terms = []
    for k in range(len(coeffs) - 1, -1, -1):
        c = coeffs[k]
        if c == 0:
            continue
        if k == 0:
            terms.append(str(c))
        else:
            mon = "z" if k == 1 else "z^%d" % k
            terms.append(mon if c == 1 else "%d%s" % (c, mon))
    return "+".join(terms)


def make_case(p, i, n, coeffs, group, timeout=DEFAULT_TIMEOUT):
    return {
        "id": "p=%d,i=%d,n=%d,E=%s" % (p, i, n, e_label(coeffs)),
        "p": p, "i": i, "n": n, "E": list(coeffs), "group": group, "timeout": timeout,
    }


def all_cases(include_optional=True):
    cases = []
    for (p, i, n) in UNRAMIFIED_PIN:
        cases.append(make_case(p, i, n, [p, 1], "unramified"))
    for (p, coeffs, ins) in RAMIFIED:
        for (i, n) in ins:
            cases.append(make_case(p, i, n, coeffs, "ramified"))
    for (p, i, n) in N1_PIN:
        cases.append(make_case(p, i, n, [p, 1], "n=1"))
    for i in E_PAIR["is"]:
        for coeffs in (E_PAIR["E_a"], E_PAIR["E_b"]):
            cases.append(make_case(E_PAIR["p"], i, E_PAIR["n"], coeffs, "E-pair"))
    if include_optional:
        for (p, i, n) in OPTIONAL_PIN:
            cases.append(make_case(p, i, n, [p, 1], "optional", OPTIONAL_TIMEOUT))
    return cases


# ---------------------------------------------------------------------------
# Precision (notebook cell 2 with exact floor-log)
# ---------------------------------------------------------------------------

def ilog(p, x):
    """Exact floor(log_p(x)) for integers x >= 1, p >= 2."""
    if x < 1:
        raise ValueError("ilog needs x >= 1 (got %d)" % x)
    k, q = 0, p
    while q <= x:
        k += 1
        q *= p
    return k


def vp_int(p, m):
    if m == 0:
        raise ValueError("v_p(0)")
    v = 0
    while m % p == 0:
        m //= p
        v += 1
    return v


def notebook_precision(nygaard_exponent, p, n, i):
    L = n * i
    T = int(nygaard_exponent(p, i, n))
    loss_delta = ilog(p, L - 1)
    loss_quotient = sum(n * vp_int(p, math.factorial(q)) for q in range(p, i))
    loss_nygaard = n * ((i * (i - 1)) // 2)
    loss_primitives = T
    loss_eisenstein = 1
    N = T + loss_delta + loss_quotient + loss_nygaard + loss_primitives + loss_eisenstein
    float_delta = math.floor(math.log(L - 1, p))
    return {
        "N": N,
        "components": {
            "target_T": T,
            "loss_delta_ilog": loss_delta,
            "loss_quotient": loss_quotient,
            "loss_nygaard": loss_nygaard,
            "loss_primitives": loss_primitives,
            "loss_eisenstein": loss_eisenstein,
        },
        "N_notebook_float_log": N - loss_delta + float_delta,
    }


# ---------------------------------------------------------------------------
# Loading KZPN (worker side)
# ---------------------------------------------------------------------------

def resolve_kzpn_dir():
    env = os.environ.get("KZPN_DIR")
    if env:
        return os.path.abspath(env), "env:KZPN_DIR"
    return os.path.abspath(os.path.join(REPO_ROOT, DEFAULT_KZPN_REL)), "default:../Background/KZPN"


def read_kzpn(kzpn_dir):
    raw = {}
    for name in KZPN_FILES:
        with open(os.path.join(kzpn_dir, name), "rb") as fh:
            raw[name] = fh.read()
    sha = {name: hashlib.sha256(b).hexdigest() for name, b in raw.items()}
    return {name: b.decode("utf-8") for name, b in raw.items()}, sha


def apply_patches(text):
    for patch in PATCHES:
        count = text.count(patch["anchor"])
        if count != 1:
            raise RuntimeError(
                "patch %s: anchor occurs %d times (expected exactly 1)" % (patch["id"], count)
            )
        at = text.index(patch["anchor"])
        if at > 0 and text[at - 1] != "\n":
            raise RuntimeError("patch %s: anchor does not start a line" % patch["id"])
        text = text.replace(patch["anchor"], patch["insert_before_anchor"] + patch["anchor"])
    return text


def load_kzpn(kzpn_dir):
    from sage.repl.preparse import preparse_file
    from sage.repl.user_globals import set_globals

    texts, sha = read_kzpn(kzpn_dir)
    for modname in ("precision", "matrix_functions"):
        mod = types.ModuleType(modname)
        mod.__file__ = "<in-memory KZPN/%s.py>" % modname
        exec(compile(texts[modname + ".py"], mod.__file__, "exec"), mod.__dict__)
        sys.modules[modname] = mod
    patched = apply_patches(texts["prismatic_envelope.sage"])
    code = preparse_file(patched)
    G = {"__name__": "kzpn_prismatic_envelope"}
    exec("from sage.all import *", G)
    set_globals(G)
    exec(compile(code, "<in-memory KZPN/prismatic_envelope.sage (patched, preparsed)>", "exec"), G)
    return G, sha, hashlib.sha256(patched.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Serialization helpers (worker side)
# ---------------------------------------------------------------------------

def enc_padic(x, p, N):
    """Encode a p-adic x as ([decimal string], absolute precision)."""
    from sage.all import ZZ, infinity

    prec = x.precision_absolute()
    if prec is infinity or prec == infinity:
        prec = N
    prec = int(prec)
    if x.is_zero():
        rep = 0
    else:
        if x.valuation() < 0:
            raise ValueError("non-integral p-adic %s" % x)
        rep = ZZ(x.lift()) % ZZ(p) ** prec
    return [str(rep)], prec


def enc_matrix(M, p, N, convention):
    rows, cols = M.nrows(), M.ncols()
    entries, precs = [], []
    for r in range(rows):
        er, pr = [], []
        for c in range(cols):
            e, pc = enc_padic(M[r, c], p, N)
            er.append(e)
            pr.append(pc)
        entries.append(er)
        precs.append(pr)
    return {"rows": rows, "cols": cols, "index_convention": convention, "entries": entries, "prec": precs}


def min_prec_all(M):
    """Minimum absolute precision over ALL entries (Sage's matrix_precision skips zero entries)."""
    return int(min(M[r, c].precision_absolute() for r in range(M.nrows()) for c in range(M.ncols())))


# ---------------------------------------------------------------------------
# Structural checks (worker side). These test the plan's matrix invariants on
# Sage's output; they are informational and never change the fixture data.
# ---------------------------------------------------------------------------

def structural_checks(G, mats, p, n, i):
    from sage.all import ZZ, factorial

    L = n * i
    eps = G["epsilon"]
    braces = G["braces"]

    def vp(m):
        return ZZ(m).valuation(p)

    def upper_nonzero(M):
        return sum(
            1 for r in range(M.nrows()) for c in range(r + 1, M.ncols()) if not M[r, c].is_zero()
        )

    def diag_mismatch(M, expected):
        bad = []
        for k, ev in enumerate(expected):
            x = M[k, k]
            got = None if x.is_zero() else int(x.valuation())
            if got != ev:
                bad.append({"index": k, "expected_valuation": ev, "got_valuation": got})
        return bad

    js = list(range(1, L))           # F-weights 1..L-1
    ms = list(range(0, L - 1))       # F-weights 0..L-2
    expected = {
        "can0": [i - j // n for j in js],
        "can1": [i - 1 - m // n for m in ms],
        "red": [vp(factorial(j // n)) for j in js],
        "redN": [vp(factorial(m // n)) for m in ms],
        "nablaOK": [vp(j) for j in js],
        "nablaR": [vp(braces(j, n)) for j in js],
        "nablaN": [int(eps(i, j, n)) + vp(braces(j, n)) for j in js],
        "canmphi": [i - j // n for j in js],
        "canmphiN": [i - 1 - (j - 1) // n for j in js],
    }
    out = {"lower_triangular": {}, "diagonal_valuations": {}}
    for name in ("can0", "can1", "red", "redN", "nablaOK", "nablaR", "nablaN", "canmphi", "canmphiN", "phi0"):
        out["lower_triangular"][name] = upper_nonzero(mats[name]) == 0
    for name, ev in expected.items():
        bad = diag_mismatch(mats[name], ev)
        out["diagonal_valuations"][name] = {"ok": not bad, "mismatches": bad[:10]}
    phi0 = mats["phi0"]
    viol = [
        [m, j]
        for m in range(1, L)
        for j in range(1, L)
        if not phi0[m - 1, j - 1].is_zero() and m < p * j
    ]
    out["phi0_support_rows_ge_p_times_col"] = {"ok": not viol, "violations": viol[:10]}
    comm = mats["nablaR"] * mats["canmphi"] - mats["canmphiN"] * mats["nablaN"]
    out["d1_d0_zero_to_precision"] = all(
        comm[r, c].is_zero() for r in range(comm.nrows()) for c in range(comm.ncols())
    )
    return out


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def worker(case, out_path):
    from copy import copy

    kzpn_dir, _ = resolve_kzpn_dir()
    rec = {k: case[k] for k in ("id", "p", "i", "n", "group")}
    rec["f"] = 1
    try:
        G, sha, patched_sha = load_kzpn(kzpn_dir)
        rec["_kzpn_sha256"] = sha
        rec["_patched_sha256"] = patched_sha
        from sage.all import ZZ, Zp, PowerSeriesRing

        p, n, i = ZZ(case["p"]), ZZ(case["n"]), ZZ(case["i"])
        L = n * i
        prec = notebook_precision(G["nygaard_exponent"], int(p), int(n), int(i))
        N = prec["N"]
        rec.update({"L": int(L), "e": len(case["E"]) - 1, "h_tilde": ["0", "1"]})
        rec["E_label"] = e_label(case["E"])
        rec["E"] = [[str(c % int(p) ** N)] for c in case["E"]]
        rec["N"] = N
        rec["N_policy"] = "notebook-cell2-exact-ilog"
        rec["N_components"] = prec["components"]
        rec["N_notebook_float_log"] = prec["N_notebook_float_log"]

        # Globals exactly as the notebook sets them (cells 1, 2, 5, 6).
        W = Zp(p, N, type="capped-abs", print_mode="terse", show_prec=False)
        A = PowerSeriesRing(W, L, names=("z",))
        (z,) = A._first_ngens(1)
        assert A.default_prec() == L
        G.update({"p": p, "i": i, "n": n, "k": n, "W": W, "A": A, "z": z, "Fprec": L,
                  "total_precision": N})
        E_poly = sum(ZZ(c) * z ** k for k, c in enumerate(case["E"]))
        E = G["eisenstein_normalization"](p, A(E_poly))
        G["E"] = E
        En, Enp = [], []
        for c in E.list():
            v, pc = enc_padic(c, p, N)
            En.append(v)
            Enp.append(pc)
        rec["E_normalized"] = En
        rec["E_normalized_prec"] = Enp
        rec["E_normalized_parent"] = str(E.parent())

        captured = {}

        def hook(**kw):
            captured.update({k: copy(v) for k, v in kw.items()})

        G["_KZPN_CAPTURE_HOOK"] = hook
        t0 = time.perf_counter()
        syn0, syn1, nablaN, nablaP = G["syntomic_matrices"](p, i, n, E, N, L)
        t1 = time.perf_counter()
        coh, minprec = G["syntomic_cohomology"](syn0, syn1, nablaN, nablaP)
        t2 = time.perf_counter()
        # Guard against global-state drift: the envelopes read globals n and i.
        assert G["n"] == n and G["i"] == i and G["p"] == p, "global p/i/n changed during the run"
        if set(captured) != {"can0", "phi0", "can1", "red", "redN", "nablaOK"}:
            raise RuntimeError("capture hook did not fire exactly as expected: %s" % sorted(captured))

        mats = dict(captured)
        mats.update({"nablaR": nablaP, "nablaN": nablaN, "canmphi": syn0, "canmphiN": syn1})

        def exps(pair):
            zero_count, torsion = pair
            vals = sorted(int(d.valuation()) for d in torsion)
            return int(zero_count), [v for v in vals if v > 0], [v for v in vals if v == 0]

        z1, h1, u1 = exps(coh["h1"])
        z2, h2, u2 = exps(coh["h2"])
        rec["h1"] = h1
        rec["h2"] = h2
        rec["h1_zero_elementary_divisors"] = z1
        rec["h2_zero_elementary_divisors"] = z2
        if u1 or u2:
            rec["unit_non_one_elementary_divisors"] = {"h1": len(u1), "h2": len(u2)}
        rec["sage_min_precision"] = int(minprec)
        rec["min_entry_precision_all"] = min(
            min_prec_all(M) for M in (syn0, syn1, nablaN, nablaP)
        )
        rec["sage_seconds"] = round(t2 - t0, 3)
        rec["sage_seconds_matrices"] = round(t1 - t0, 3)
        rec["sage_seconds_cohomology"] = round(t2 - t1, 3)
        rec["min_entry_precision_d0"] = min(min_prec_all(syn0), min_prec_all(nablaN))
        rec["min_entry_precision_d1"] = min(min_prec_all(nablaP), min_prec_all(syn1))
        rec["bookkeeping_ok"] = (sum(h1) - sum(h2) == int(i) * (int(n) - 1))
        rec["h1_exponents_below_d0_precision"] = all(a < rec["min_entry_precision_d0"] for a in h1)
        rec["h2_exponents_below_d1_precision"] = all(a < rec["min_entry_precision_d1"] for a in h2)
        rec["checks"] = structural_checks(G, mats, int(p), int(n), int(i))
        order = ["can0", "phi0", "can1", "red", "redN", "nablaOK", "nablaR", "nablaN", "canmphi", "canmphiN"]
        rec["matrices"] = {
            name: enc_matrix(mats[name], int(p), N, INDEX_CONVENTIONS[name]) for name in order
        }
        rec["error"] = None
    except Exception as exc:  # recorded, never fatal for the whole run
        rec["error"] = "%s: %s" % (type(exc).__name__, exc)
        rec["traceback"] = traceback.format_exc()
    with open(out_path, "w") as fh:
        json.dump(rec, fh)


# ---------------------------------------------------------------------------
# Parent / orchestration
# ---------------------------------------------------------------------------

def run_one(case, tmpdir):
    out_path = os.path.join(tmpdir, "case_%s.json" % hashlib.sha1(case["id"].encode()).hexdigest())
    cmd = [sys.executable, SCRIPT_PATH, "--worker", json.dumps(case), out_path]
    t0 = time.perf_counter()
    try:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=case["timeout"], env=env)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "case": case, "wall": time.perf_counter() - t0}
    wall = time.perf_counter() - t0
    if not os.path.exists(out_path):
        return {"status": "crash", "case": case, "wall": wall,
                "stderr": proc.stderr[-4000:], "returncode": proc.returncode}
    with open(out_path) as fh:
        rec = json.load(fh)
    return {"status": "ok", "case": case, "wall": wall, "record": rec, "stderr": proc.stderr[-2000:]}


def dump_json(obj_top, cases, path):
    """Pretty top-level, one compact line per case (keeps big fixtures diff-friendly)."""
    lines = ["{"]
    keys = list(obj_top.keys())
    for k in keys:
        lines.append("%s: %s," % (json.dumps(k), json.dumps(obj_top[k], indent=1)))
    lines.append('"cases": [')
    for idx, c in enumerate(cases):
        sep = "," if idx + 1 < len(cases) else ""
        lines.append(json.dumps(c, separators=(",", ":")) + sep)
    lines.append("]")
    lines.append("}")
    text = "\n".join(lines) + "\n"
    json.loads(text)  # validate
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)


def parent(args):
    import sage.version

    kzpn_dir, kzpn_source = resolve_kzpn_dir()
    texts, sha = read_kzpn(kzpn_dir)
    kzpn_prov = kzpn_provenance(kzpn_dir, sha)
    patched_sha = hashlib.sha256(apply_patches(texts["prismatic_envelope.sage"]).encode("utf-8")).hexdigest()
    cases = all_cases(include_optional=not args.no_optional)
    if (args.only or args.no_optional) and os.path.abspath(args.output) == os.path.abspath(DEFAULT_OUTPUT):
        raise SystemExit("--only/--no-optional write a partial fixture; pass --output PATH so the "
                         "committed test/fixtures/sage_f1.json is not overwritten")
    if args.only:
        wanted = set(args.only)
        cases = [c for c in cases if c["id"] in wanted]
        missing = wanted - {c["id"] for c in cases}
        if missing:
            raise SystemExit("unknown case ids: %s" % sorted(missing))
    log = lambda msg: print(msg, file=sys.stderr, flush=True)
    log("KZPN dir (%s): %s" % (kzpn_source, kzpn_dir))
    warn_if_not_upstream(kzpn_prov, log)
    log("running %d cases with %d parallel workers" % (len(cases), args.jobs))

    results = {}
    with tempfile.TemporaryDirectory(prefix="kzpn_f1_") as tmpdir:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            # Longest (largest L) first; the output order is the case-list order regardless.
            ordered = sorted(cases, key=lambda c: -c["n"] * c["i"])
            futs = {pool.submit(run_one, c, tmpdir): c["id"] for c in ordered}
            for fut in futs:
                res = fut.result()
                results[futs[fut]] = res
                c = res["case"]
                if res["status"] == "ok":
                    r = res["record"]
                    if r.get("error"):
                        log("[error]   %-40s %s" % (c["id"], r["error"]))
                    else:
                        log("[ok]      %-40s N=%-4d h1=%s h2=%s  %.2fs" % (
                            c["id"], r["N"], r["h1"], r["h2"], r["sage_seconds"]))
                else:
                    log("[%s] %-40s after %.1fs" % (res["status"], c["id"], res["wall"]))

    out_cases, skipped = [], []
    for c in cases:
        res = results[c["id"]]
        if res["status"] == "timeout":
            skipped.append({"id": c["id"], "p": c["p"], "i": c["i"], "n": c["n"],
                            "E": [[str(x)] for x in c["E"]], "group": c["group"],
                            "reason": "timeout: exceeded %d s wall clock" % c["timeout"]})
            continue
        if res["status"] == "crash":
            skipped.append({"id": c["id"], "p": c["p"], "i": c["i"], "n": c["n"],
                            "E": [[str(x)] for x in c["E"]], "group": c["group"],
                            "reason": "worker crashed (return code %s)" % res["returncode"],
                            "stderr_tail": res["stderr"][-1500:]})
            continue
        r = res["record"]
        ksha = r.pop("_kzpn_sha256", None)
        psha = r.pop("_patched_sha256", None)
        if ksha is not None and (ksha != sha or psha != patched_sha):
            raise SystemExit("KZPN sources changed during the run (hash mismatch) for %s" % c["id"])
        tb = r.pop("traceback", None)
        if tb:
            log("traceback for %s:\n%s" % (c["id"], tb))
        out_cases.append(r)

    # E-pair comparison: same ring R, must give identical cohomology.
    comparisons = []
    by_id = {r["id"]: r for r in out_cases}
    for i in E_PAIR["is"]:
        ida = make_case(E_PAIR["p"], i, E_PAIR["n"], E_PAIR["E_a"], "E-pair")["id"]
        idb = make_case(E_PAIR["p"], i, E_PAIR["n"], E_PAIR["E_b"], "E-pair")["id"]
        ra, rb = by_id.get(ida), by_id.get(idb)
        entry = {"kind": "same-ring-different-E", "ids": [ida, idb],
                 "reason": "E_a = E_b mod 2^ceil(n/e) = 4, so R = Z_2[z]/(E, z^n) is the same ring"}
        if ra and rb and not ra.get("error") and not rb.get("error"):
            entry["h1"] = [ra["h1"], rb["h1"]]
            entry["h2"] = [ra["h2"], rb["h2"]]
            entry["agree"] = (ra["h1"] == rb["h1"] and ra["h2"] == rb["h2"])
        else:
            entry["agree"] = None
        comparisons.append(entry)

    top = {
        "format": FORMAT_ID,
        "provenance": {
            "trust": "sage-original",
            "generator": "scripts/sage/make_fixtures_f1.py",
            "generation_date": args.date,
            "sage_version": sage.version.version,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "kzpn_dir_source": kzpn_source,
            **kzpn_prov,
            "loader": (
                "precision.py and matrix_functions.py exec'd from source into in-memory modules registered in "
                "sys.modules; prismatic_envelope.sage patched in memory, preparsed with "
                "sage.repl.preparse.preparse_file and exec'd into a namespace G with "
                "sage.repl.user_globals.set_globals(G); bytecode writing disabled; one fresh worker process "
                "per case"
            ),
            "patch_set_id": PATCH_SET_ID,
            "patches": PATCHES,
            "patched_prismatic_envelope_sha256": patched_sha,
            "mathematical_changes": "none",
            "globals_set_per_case": ["p", "i", "n", "k", "W", "A", "z", "E", "Fprec", "total_precision"],
            "sign_convention": SIGN_CONVENTION,
            "precision_policy": PRECISION_POLICY,
            "timing_note": (
                "sage_seconds = wall clock of syntomic_matrices + syntomic_cohomology inside the worker "
                "(excludes Sage start-up and KZPN load); %d workers ran concurrently" % args.jobs
            ),
        },
        "conventions": {
            "p_adic_numbers": "decimal strings of a representative in [0, p^prec)",
            "W_element": (
                "list of f decimal strings: coordinates in the basis 1, x, ..., x^(f-1) of Z_p[x]/(h_tilde); "
                "here f = 1, h_tilde = [\"0\",\"1\"], lists of length 1"
            ),
            "polynomials": "lists low->high degree of W-elements (E, E_normalized)",
            "matrix": (
                "{rows, cols, index_convention, entries (row-major, W-elements), prec (absolute p-adic "
                "precision per entry as reported by Sage; capped-abs exact zeros report N)}"
            ),
            "cohomology": "h1/h2 = sorted exponent lists a_j with H = sum Z/p^{a_j}; [] = 0",
            "notation": "paper n = power of the uniformizer (Sage k), i = weight, L = n*i (Sage Fprec)",
            "differentials": DIFFERENTIALS,
            "fields": {
                "N": "working precision used (W = Zp(p, N, capped-abs))",
                "N_notebook_float_log": "N the notebook's float math.log would have given (informational)",
                "sage_min_precision": "second return value of syntomic_cohomology (skips zero entries)",
                "min_entry_precision_all": "min absolute precision over all entries of canmphi, canmphiN, nablaN, nablaR",
                "min_entry_precision_d0": "min absolute precision over all entries of d0 = [canmphi; nablaN]",
                "min_entry_precision_d1": "min absolute precision over all entries of d1 = [nablaR, -canmphiN]",
                "h1_exponents_below_d0_precision": "every H^1 exponent < min_entry_precision_d0 (Remark 7.3-style check)",
                "h2_exponents_below_d1_precision": "every H^2 exponent < min_entry_precision_d1",
                "h1_zero_elementary_divisors": "number of elementary divisors of d0 that are 0 to precision (should be 0)",
                "h2_zero_elementary_divisors": "number of elementary divisors of d1 that are 0 to precision (should be 0)",
                "bookkeeping_ok": "sum(h1) - sum(h2) == i*(n-1) (Prop 2.18 with f = 1)",
                "checks": "plan invariants evaluated on Sage's matrices (informational only)",
            },
            "index_conventions": INDEX_CONVENTIONS,
        },
        "comparisons": comparisons,
        "skipped": skipped,
    }
    dump_json(top, out_cases, args.output)
    log("wrote %s (%d cases, %d skipped)" % (args.output, len(out_cases), len(skipped)))
    for cmp_ in comparisons:
        log("E-pair %s: agree=%s" % (cmp_["ids"], cmp_["agree"]))


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        worker(json.loads(sys.argv[2]), sys.argv[3])
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=datetime.date.today().isoformat(),
                    help="generation date recorded in provenance (string)")
    ap.add_argument("--jobs", type=int, default=4, help="parallel worker processes")
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--only", nargs="*", help="run only these case ids")
    ap.add_argument("--no-optional", action="store_true", help="skip the optional (3,8,4) case")
    ap.add_argument("--list", action="store_true", help="print the case ids and exit")
    args = ap.parse_args()
    if args.list:
        for c in all_cases(True):
            print(c["id"])
        return
    parent(args)


if __name__ == "__main__":
    main()
