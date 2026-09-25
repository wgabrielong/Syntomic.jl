#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
make_fixtures_fq.py -- f > 1 fixtures for Syntomic.jl from a SIGMA-PATCHED,
in-memory copy of the AKN/KZPN Sage code (trust level "sage-sigma-patched").

Run (Sage 10.3):

    sage -python scripts/sage/make_fixtures_fq.py --date YYYY-MM-DD [--out PATH]
                 [--only CASE_ID ...] [--margin K] [--no-julia-check]

Environment:
    KZPN_DIR   directory holding prismatic_envelope.sage, precision.py and
               matrix_functions.py (default: <repo root>/../Background/KZPN).

What it does
------------
* Reads the KZPN sources as text (nothing is imported from, or written next to,
  the KZPN directory), applies anchored text patches to the .sage source (each
  anchor must occur exactly once), preparses it in memory with
  sage.repl.preparse.preparse_file and exec's it into a private namespace G.
  precision.py and matrix_functions.py are exec'd into the same namespace.
  sage.repl.user_globals.set_globals(G) makes inject_variables write into G.
* The patches:
  sigma on coefficients for every Frobenius V(p) site (L127, L249, L331, L482,
  and E(z^p) at L482), integer base-p digits instead of W(n//k)[j], and
  workarounds for the Sage 10.3 bugs of power series over Zq.
* Per case it sets the globals p, i, n, W, A, z (the KZPN code reads the global
  n and i inside prismatic_envelope_f/g), runs the truncated syntomic_matrices
  (which returns the raw W-matrices can0, phi0, can1, red, redN, nablaOK),
  expands them to Z_p-matrices (W-linear A -> M(A); sigma-semilinear
  Phi -> M(Phi)(I kron S); the index of x^d*b_m is f*(m-1)+d+1), performs the
  square completions of Steps 7, 8, 10 over Zp capped-abs and computes the
  elementary divisors of d0 = [can-phi ; N nabla] and d1 = [nabla, -(can-phi^nabla)].
* Self-checks: brute-force i=1 values, Hesselholt-Madsen formula
  (n <= e), Angeltveit W_{i(n-1)}(F_q) (p=5, i <= p-2), invariance families
  (E <-> E^sigma, E(z) <-> E(az), E <-> E + p^{ceil(n/e)} h), the order
  bookkeeping sum(H1) - sum(H2) = f*i*(n-1), an independent integer Smith-form
  certificate, and a rerun at higher working precision.

Sign convention: Sage's g0 = z0 - z1 (eta_R(z) = z - g0).  For odd p the
nablaOK matrix equals the paper-convention (z1 = z0 + g0) matrix exactly; for
p = 2 it differs by units, so compare cohomology only there.

Output: <repo>/test/fixtures/sage_fq.json (see the "conventions" object inside).
"""

import sys

sys.dont_write_bytecode = True  # never write __pycache__ anywhere

import argparse
import hashlib
import json
import os
import subprocess
import time
import traceback

from sage.all import (ZZ, Zq, Integer, Matrix, PolynomialRing, PowerSeriesRing,
                      binomial, block_matrix, factorial, identity_matrix,
                      valuation, version)
from sage.repl.preparse import preparse_file
from sage.repl.user_globals import set_globals
import sage.version as _sage_version
from sage.rings.polynomial.polydict import ETuple
from sage.rings.polynomial.multi_polynomial_element import MPolynomial_polydict
from cysignals.alarm import alarm, cancel_alarm, AlarmInterrupt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_KZPN = os.path.join(REPO_ROOT, "..", "Background", "KZPN")
DEFAULT_OUT = os.path.join(REPO_ROOT, "test", "fixtures", "sage_fq.json")
KZPN_FILES = ["prismatic_envelope.sage", "precision.py", "matrix_functions.py"]
PATCH_SET_ID = "kzpn-sigma-fq-v1"
CASE_TIMEOUT_S = 120

# ---------------------------------------------------------------------------
# Anchored text patches for prismatic_envelope.sage (applied in this order).
# kind "replace": `old` must occur exactly once; kind "cut": `start` and `end`
# must each occur exactly once, and the text [start, end) is deleted.
# ---------------------------------------------------------------------------

PATCHES = [
    dict(id="K00-imports", kind="replace", site="L1-2",
         why="precision.py and matrix_functions.py are exec'd into the namespace by the "
             "harness (no import from, and no bytecode next to, the KZPN directory)",
         old="from precision import *\nfrom matrix_functions import *\n",
         new="# [K00] precision.py and matrix_functions.py exec'd into this namespace by the harness\n"),
    dict(id="K01-drop-v1", kind="cut", site="L746-1022",
         why="syntomic_matrices_v1 is flagged 'not correct' upstream and unused; removing it "
             "makes the digit-site anchors below unique",
         start="def syntomic_matrices_v1(p,i,k,E,prec,Fprec,debug=False):\n",
         end="###########################################################################\n"
             "# Compute the cohomology of the syntomic complex via elementary divisors. #"),
    # --- sigma: Frobenius on W[[z]] is sigma on coefficients followed by z -> z^p ---
    dict(id="S01-frob-L127", kind="replace", site="L127 (f-side phitilde)",
         why="Frobenius = sigma on coefficients then z->z^p",
         old="g+=c.V(p)*(m(fvars_phitilde))",
         new="g+=FROB(c)*(m(fvars_phitilde))"),
    dict(id="S02-frob-L249", kind="replace", site="L249 (f-side phi_divided)",
         why="Frobenius = sigma on coefficients then z->z^p",
         old="result+=B(c.V(p)*m(fvars_phi_divided))",
         new="result+=B(FROB(c)*m(fvars_phi_divided))"),
    dict(id="S03-frob-L331", kind="replace", site="L331 (g-side phitilde)",
         why="Frobenius = sigma on coefficients then z->z^p",
         old="h+=x.V(p)*n^p",
         new="h+=FROB(x)*n^p"),
    dict(id="S04-frob-L482", kind="replace", site="L482 (g-side phi, incl. E(z^p))",
         why="Frobenius = sigma on coefficients then z->z^p; E(z^p) -> phi(E) = E^sigma(z^p)",
         old="h+=x.V(p)*C(E(z^p))^deg",
         new="h+=FROB(x)*C(FROB(A(E)))^deg"),
    # --- integer base-p digits instead of W(m//k)[j] (Zq returns coordinate lists) ---
    dict(id="D01-digits-can0-col", kind="replace", site="L633",
         why="integer base-p digits (W(n//k)[j] is a coordinate list over Zq)",
         old="        d=W(n//k)\n        fprod=B(1)\n        for j in range(num_f):\n"
             "            fprod=fprod*fvars[j]^(d[j])\n"
             "        column_to_process=recreduceB(E(z)^(i-(n//k))*z^c*fprod)",
         new="        d=DIG(n//k)\n        fprod=B(1)\n        for j in range(num_f):\n"
             "            fprod=fprod*fvars[j]^(d[j])\n"
             "        column_to_process=recreduceB(E(z)^(i-(n//k))*z^c*fprod)"),
    dict(id="D02-digits-phi0-col", kind="replace", site="L658",
         why="integer base-p digits",
         old="        d=W(n//k)\n        fprod=B(1)\n        for j in range(num_f):\n"
             "            fprod=fprod*fvars[j]^(d[j])\n"
             "        column_to_process=recreduceB(z^(p*c)*phi_dividedB(fprod))",
         new="        d=DIG(n//k)\n        fprod=B(1)\n        for j in range(num_f):\n"
             "            fprod=fprod*fvars[j]^(d[j])\n"
             "        column_to_process=recreduceB(z^(p*c)*phi_dividedB(fprod))"),
    dict(id="D03-digits-can1-col", kind="replace", site="L680",
         why="integer base-p digits",
         old="        d=W(n//k)\n        fprod=B(1)\n        for j in range(num_f):\n"
             "            fprod=fprod*fvars[j]^(d[j])\n"
             "        column_to_process=recreduceB(E(z)^(i-(n//k)-1)*z^c*fprod)",
         new="        d=DIG(n//k)\n        fprod=B(1)\n        for j in range(num_f):\n"
             "            fprod=fprod*fvars[j]^(d[j])\n"
             "        column_to_process=recreduceB(E(z)^(i-(n//k)-1)*z^c*fprod)"),
]

_ROW_TAIL = ("            gprod=B(1)\n            for j in range(num_f):\n"
             "                gprod=gprod*fvars[j]^(b[j])\n"
             "            coefficient_to_process=column_to_process.monomial_coefficient(gprod)\n")
for _pid, _site, _target in [("D04-digits-can0-row", "L641", "            can0[m-1,n-1]=coefficient_to_process[a]"),
                             ("D05-digits-phi0-row", "L666", "            phi0[m-1,n-1]=coefficient_to_process[a]"),
                             ("D06-digits-can1-row", "L688", "            can1[m,n]=coefficient_to_process[a]"),
                             ("D07-digits-red-row", "L712", "            OKtoOKmodpi0[m-1,n-1]=coefficient_to_process[a]"),
                             ("D08-digits-redN-row", "L725", "            OKtoOKmodpi1[m,n]=coefficient_to_process[a]")]:
    PATCHES.append(dict(id=_pid, kind="replace", site=_site, why="integer base-p digits",
                        old="b=W(m//k)\n" + _ROW_TAIL + _target,
                        new="b=DIG(m//k)\n" + _ROW_TAIL + _target))

PATCHES += [
    # --- Sage 10.3 Zq workarounds ---
    dict(id="Q01-fvars-gen", kind="replace", site="L62-63",
         why="B('f') string coercion fails over Zq power series; use B.gen(j)",
         old="    for j in variable_names:\n        fvars.append(B(j))",
         new="    for j in range(len(variable_names)):\n        fvars.append(B.gen(j))"),
    dict(id="Q02-gvars-gen", kind="replace", site="L275-276",
         why="C('g') string coercion fails over Zq power series; use C.gen(j)",
         old="    for j in variable_names:\n        gvars.append(C(j))",
         new="    for j in range(len(variable_names)):\n        gvars.append(C.gen(j))"),
    dict(id="Q03-numf-f", kind="replace", site="L53-54",
         why="a polynomial ring in zero variables over Zq power series breaks; num_f=0 -> 1 "
             "(harmless: f0 has F-weight n = L when i = 1, so it is always trimmed)",
         old="    if i==1:\n        num_f=0\n    else:\n        num_f=floor(log(i-1,p))+1\n"
             "    B=PolynomialRing(A,'f',num_f)",
         new="    if i==1:\n        num_f=1\n    else:\n        num_f=floor(log(i-1,p))+1\n"
             "    B=PolynomialRing(A,'f',num_f)"),
    dict(id="Q04-numf-matrices", kind="replace", site="L621-622",
         why="keep num_f consistent with Q03 inside syntomic_matrices",
         old="    if i==1:\n        num_f=0\n    else:\n        num_f=floor(log(i-1,p))+1\n\n    # Basis z^c",
         new="    if i==1:\n        num_f=1\n    else:\n        num_f=floor(log(i-1,p))+1\n\n    # Basis z^c"),
    dict(id="Q05-fixpoint-L201", kind="replace", site="L201",
         why="== / != on Zq power series is broken (A(1)==A(1) is False); coefficientwise "
             "inexact-zero test",
         old="new_rels[j]=reduce(B(prev_value))\n        while prev_value != new_rels[j]:",
         new="new_rels[j]=reduce(B(prev_value))\n        while not ISZERO(prev_value - new_rels[j]):"),
    dict(id="Q06-fixpoint-L238", kind="replace", site="L238",
         why="coefficientwise inexact-zero fixed-point test",
         old="while prev_value!=funct:",
         new="while not ISZERO(prev_value-funct):"),
    dict(id="Q07-fixpoint-L444", kind="replace", site="L444",
         why="coefficientwise inexact-zero fixed-point test",
         old="new_rels[j]=reduce(C(prev_value))\n        while prev_value != new_rels[j]:",
         new="new_rels[j]=reduce(C(prev_value))\n        while not ISZERO(prev_value - new_rels[j]):"),
    dict(id="Q08-fixpoint-L458", kind="replace", site="L458",
         why="coefficientwise inexact-zero fixed-point test",
         old="while prevFunct != funct:",
         new="while not ISZERO(prevFunct - funct):"),
    dict(id="Q09-fixpoint-L539", kind="replace", site="L539",
         why="coefficientwise inexact-zero fixed-point test",
         old="if funct==v:",
         new="if ISZERO(funct-v):"),
    dict(id="Q10-trim-f", kind="replace", site="L148-153",
         why="PolyDict does not drop inexact zeros over Zq: drop monomials with F-weight >= Fprec "
             "or with an inexactly-zero coefficient",
         old="        g=0\n        for m in funct.monomials():\n            w=weight(m)[0]\n"
             "            new_coefficient=funct.monomial_coefficient(m).add_bigoh(max(Fprec-w,0))\n"
             "            g+=new_coefficient*m\n        return B(g)",
         new="        g=0\n        for m in funct.monomials():\n            w=weight(m)[0]\n"
             "            if w>=Fprec:\n                continue\n"
             "            new_coefficient=funct.monomial_coefficient(m).add_bigoh(max(Fprec-w,0))\n"
             "            if ISZERO(new_coefficient):\n                continue\n"
             "            g+=new_coefficient*m\n        return B(g)"),
    dict(id="Q11-trim-g", kind="replace", site="L344-353",
         why="as Q10 for the g-side trim",
         old="        trimmed=0\n        for m in funct.monomials():\n            w=weight(m)[0]\n"
             "            new_coefficient=0\n            c=funct.monomial_coefficient(m)\n"
             "            for n in c.monomials():\n"
             "                to_trim=c.monomial_coefficient(n).add_bigoh(max(Fprec-w,0))\n"
             "                new_coefficient+=to_trim*n\n            trimmed+=new_coefficient*m\n"
             "        return C(trimmed)",
         new="        trimmed=0\n        for m in funct.monomials():\n            w=weight(m)[0]\n"
             "            if w>=Fprec:\n                continue\n"
             "            new_coefficient=0\n            c=funct.monomial_coefficient(m)\n"
             "            for n in c.monomials():\n"
             "                to_trim=c.monomial_coefficient(n).add_bigoh(max(Fprec-w,0))\n"
             "                if ISZERO(to_trim):\n                    continue\n"
             "                new_coefficient+=to_trim*n\n"
             "            if ISZERO(new_coefficient):\n                continue\n"
             "            trimmed+=new_coefficient*m\n"
             "        return C(trimmed)"),
    dict(id="Q12-g0-divide", kind="replace", site="L494-495",
         why="skip g0-free monomials whose coefficient is inexactly zero",
         old="            if degs[0]==0:\n                raise TypeError('Cannot divide by g0.')",
         new="            if degs[0]==0:\n                if ISZERO(funct.monomial_coefficient(m)):\n"
             "                    continue\n                raise TypeError('Cannot divide by g0.')"),
    dict(id="Q13-g0-coefficient", kind="replace", site="L594-597",
         why="dict-keyed .coefficient hashes Zq elements (unhashable); look up ETuple (1,0,...,0)",
         old="            send_g0_to_one_dict={gvarsC[0]:1}\n"
             "            for j in range(1,len(gvarsC)):\n"
             "                send_g0_to_one_dict[gvarsC[j]]=0\n"
             "            send_g0_to_one=A(column_to_process.coefficient(send_g0_to_one_dict))",
         new="            send_g0_to_one=G0COEF(column_to_process,len(gvarsC))"),
    # --- driver: stop before the W-level square completions ---
    dict(id="R01-return-W-matrices", kind="replace", site="before L732",
         why="return raw W-matrices; the harness expands to Z_p and completes the squares there",
         old="    # Given nablaP for OK, compute it for OK/pi^k.\n",
         new="    return can0,phi0,can1,OKtoOKmodpi0,OKtoOKmodpi1,nablaP_OK  # [R01] harness\n"
             "    # Given nablaP for OK, compute it for OK/pi^k.\n"),
]

HARNESS_NOTES = [
    "Before every call the namespace globals p, i, n, W, A, z (plus a, Fprec = L, k = n) are set for the case; "
    "the KZPN code reads the global n and i inside prismatic_envelope_f/g.",
    "W = Zq(p^f, N, names='a', type='capped-abs', print_mode='series'); A = PowerSeriesRing(W, 'z', L). "
    "Zq's default modulus is the Conway polynomial (verified against conway_polynomial and Nemo/Hecke).",
    "E is passed as A(eisenstein_normalization(p, E)): the normalization returns a series over Frac(W) whose "
    "coercion into B = A[f_0, ...] fails over Zq in Sage 10.3; the normalized E is integral, so this is exact.",
    "FROB(c) = (sigma on each coefficient via Zq .frobenius(), same z-precision).V(p); DIG(t) = ZZ(t).digits(p, padto=12); "
    "ISZERO = every Zq coefficient is_zero() (inexact zero); G0COEF = lookup of ETuple (1,0,...,0) in .dict().",
    "Square completions after Z_p expansion with KZPN matrix_functions.square_complete_bottom/top/right over "
    "Zp(p, N) capped-abs; H1/H2 from Sage p-adic elementary divisors, cross-checked with an integer Smith form "
    "of [lift(d) | p^M I] (M = min absolute precision of d).",
    "Negative controls rerun selected cases with FROB(c) = c.V(p) (upstream, no sigma) and S = I.",
]

MONKEYPATCHES = [
    dict(id="M01-monomial-coefficient",
         target="sage.rings.polynomial.multi_polynomial_element.MPolynomial_polydict.monomial_coefficient",
         why="raises 'not a monomial' over Zq power series (its is-one test fails); replaced by an "
             "exponent lookup in self.dict()"),
]

# Helpers exec'd into the KZPN namespace G (so that p, A are G's globals).
HELPERS = r'''
from sage.rings.power_series_ring_element import PowerSeries as _PS
from sage.rings.polynomial.polynomial_element import Polynomial as _UPoly
from sage.rings.polynomial.multi_polynomial import MPolynomial as _MPoly
from sage.rings.polynomial.polydict import ETuple as _ETuple

def ISZERO(x):
    """Coefficientwise inexact-zero test (== is broken for power series over Zq)."""
    if isinstance(x, _PS):
        return all(t.is_zero() for t in x.list())
    if isinstance(x, _MPoly):
        return all(ISZERO(t) for t in x.dict().values())
    if isinstance(x, _UPoly):
        return all(ISZERO(t) for t in x.list())
    if isinstance(x, int):
        return x == 0
    return x.is_zero()

def FROB(c):
    """Frobenius lift on W[[z]]: sigma on the coefficients, then z -> z^p."""
    if not isinstance(c, _PS):
        c = A(c)
    P = c.parent()
    cs = [t.frobenius() for t in c.list()]
    pr = c.prec()
    s = P(cs) if pr == Infinity else P(cs, pr)
    return s.V(p)

def DIG(t):
    """Integer base-p digits of t (replaces W(t)[j], which is only digits for f = 1)."""
    return ZZ(t).digits(p, padto=12)

def G0COEF(col, ng):
    """A-coefficient of the monomial g0 (other g_u exponents 0) of a reduced element."""
    c = col.dict().get(_ETuple([1] + [0]*(ng-1)), None)
    if c is None:
        return A(0)
    cl = c.list()
    for t in cl[1:]:
        if not ISZERO(t):
            raise ValueError("g0-coefficient is not d-free after reduction")
    return A(cl[0]) if cl else A(0)
'''


def _monomial_coefficient(self, mon):
    md = mon.dict()
    if len(md) != 1:
        raise ValueError("not a monomial (harness M01)")
    (exp,) = list(md.keys())
    return self.dict().get(exp, self.parent().base_ring().zero())


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def apply_patches(src):
    applied = []
    for pt in PATCHES:
        if pt["kind"] == "replace":
            cnt = src.count(pt["old"])
            if cnt != 1:
                raise RuntimeError("patch %s: anchor occurs %d times (expected 1)" % (pt["id"], cnt))
            src = src.replace(pt["old"], pt["new"])
        elif pt["kind"] == "cut":
            for key in ("start", "end"):
                cnt = src.count(pt[key])
                if cnt != 1:
                    raise RuntimeError("patch %s: %s anchor occurs %d times" % (pt["id"], key, cnt))
            a = src.index(pt["start"])
            b = src.index(pt["end"])
            if not a < b:
                raise RuntimeError("patch %s: start after end" % pt["id"])
            src = src[:a] + "# [K01] syntomic_matrices_v1 removed by harness\n\n" + src[b:]
        else:
            raise RuntimeError("unknown patch kind")
        applied.append(dict(id=pt["id"], kind=pt["kind"], site=pt["site"], why=pt["why"]))
    return src, applied


def load_kzpn(kzpn_dir):
    texts = {}
    for fn in KZPN_FILES:
        with open(os.path.join(kzpn_dir, fn), "r") as fh:
            texts[fn] = fh.read()
    patched, applied = apply_patches(texts["prismatic_envelope.sage"])
    MPolynomial_polydict.monomial_coefficient = _monomial_coefficient
    G = {"__name__": "kzpn_sigma_patched"}
    exec("from sage.all import *", G)
    exec(compile(texts["precision.py"], "<KZPN precision.py>", "exec"), G)
    exec(compile(texts["matrix_functions.py"], "<KZPN matrix_functions.py>", "exec"), G)
    exec(compile(HELPERS, "<harness helpers>", "exec"), G)
    set_globals(G)
    py = preparse_file(patched)
    exec(compile(py, "<KZPN prismatic_envelope.sage [sigma-patched]>", "exec"), G)
    return G, applied


# ---------------------------------------------------------------------------
# Exact arithmetic in Z[x]/(h) for the case definitions (E^sigma, E(az))
# ---------------------------------------------------------------------------

class ExactGR(object):
    """Z[x]/(h), h monic (the Conway lift): exact integer coordinates."""

    def __init__(self, p, f, h):
        self.p, self.f = p, f
        self.Zx = PolynomialRing(ZZ, "x")
        self.x = self.Zx.gen()
        self.h = self.Zx(list(h))
        assert self.h.is_monic() and self.h.degree() == f

    def elt(self, coords):
        return self.Zx(list(coords)) % self.h

    def coords(self, e):
        c = [ZZ(t) for t in (e % self.h).list()]
        return c + [ZZ(0)] * (self.f - len(c))

    def teichmuller_generator(self):
        q = self.p ** self.f
        return ((self.x ** q - self.x) % self.h) == 0

    def sigma(self, e):
        # exact only when x is Teichmueller (x^q = x); then sigma(x) = x^p
        assert self.teichmuller_generator(), "sigma(x) = x^p is exact only for a Teichmueller generator"
        return self.Zx((e % self.h)(self.x ** self.p)) % self.h

    def poly_sigma(self, E):
        return [self.coords(self.sigma(self.elt(c))) for c in E]

    def poly_scale(self, E, u):
        """E(u z) for u in Z[x]/(h)."""
        ue = self.elt(u)
        out, pw = [], self.Zx(1)
        for c in E:
            out.append(self.coords(self.elt(c) * pw))
            pw = (pw * ue) % self.h
        return out

    def poly_add_multiple(self, E, k, c):
        """E + c z^k (c exact coords)."""
        E = [list(t) for t in E]
        while len(E) <= k:
            E.append([0] * self.f)
        E[k] = self.coords(self.elt(E[k]) + self.elt(c))
        return E


def fmt_welt(coords):
    """Human-readable a-polynomial."""
    terms = []
    for d, c in enumerate(coords):
        c = ZZ(c)
        if c == 0:
            continue
        mon = "" if d == 0 else ("a" if d == 1 else "a^%d" % d)
        if mon == "":
            terms.append(str(c))
        elif c == 1:
            terms.append(mon)
        elif c == -1:
            terms.append("-" + mon)
        else:
            terms.append("%s*%s" % (c, mon))
    if not terms:
        return "0"
    s = "+".join(terms).replace("+-", "-")
    return s


def fmt_poly(E):
    terms = []
    for k in range(len(E) - 1, -1, -1):
        cs = fmt_welt(E[k])
        if cs == "0":
            continue
        zm = "" if k == 0 else ("z" if k == 1 else "z^%d" % k)
        if zm == "":
            terms.append(cs)
        elif cs == "1":
            terms.append(zm)
        elif cs == "-1":
            terms.append("-" + zm)
        else:
            multi = ("+" in cs[1:]) or ("-" in cs[1:])
            terms.append(("(%s)" % cs if multi else cs) + "*" + zm)
    return "+".join(terms).replace("+-", "-")


# ---------------------------------------------------------------------------
# Case list
# ---------------------------------------------------------------------------

# Conway lifts (checked at run time against Sage's Zq modulus, Sage's
# conway_polynomial and Nemo/Hecke defining_polynomial(qadic_field(p, f))).
CONWAY = {(2, 2): [1, 1, 1], (3, 2): [2, 2, 1], (2, 3): [1, 1, 0, 1], (5, 2): [2, 4, 1]}

# Reference values (H1, H2 exponent lists).
BRUTE_FORCE_R12A = "brute-force enumeration of U^1/U^n"
HM_R12B = "Hesselholt-Madsen formula"
INV_R12C = "invariance under isomorphic presentations"
ANG_R12D = "Angeltveit W_{i(n-1)}(F_q) structure"


def build_cases():
    cases = []
    families = []

    def pad(E, f):
        return [[ZZ(t) for t in c] + [ZZ(0)] * (f - len(c)) for c in E]

    def add(cid, p, f, n, i, E, tags, refs=None, derived=None, sigma_of=None):
        cases.append(dict(id=cid, p=p, f=f, n=n, i=i, E=pad(E, f), tags=tags, refs=refs or [], derived=derived,
                          sigma_of=pad(sigma_of, f) if sigma_of is not None else None))

    gr = {pf: ExactGR(pf[0], pf[1], CONWAY[pf]) for pf in CONWAY}

    # E encodings: list (low -> high degree) of coefficient coordinate lists in 1, a, a^2.
    z2_2 = [[2], [0], [1]]                 # z^2 + 2
    z2_2az_2 = [[2], [0, 2], [1]]          # z^2 + 2az + 2
    z2_2az_2a = [[0, 2], [0, 2], [1]]      # z^2 + 2az + 2a
    z2_3az_3 = [[3], [0, 3], [1]]          # z^2 + 3az + 3
    z2_3az_3a = [[0, 3], [0, 3], [1]]      # z^2 + 3az + 3a
    z3_2az_2 = [[2], [0, 2], [0], [1]]     # z^3 + 2az + 2
    z_2 = [[2], [1]]
    z_3 = [[3], [1]]
    z_5 = [[5], [1]]

    # (a) i = 1 against brute force
    bf = [
        ("bf_p2f2_z2+2_n3", 2, 2, 3, z2_2, [2, 2]),
        ("bf_p2f2_z2+2az+2_n3", 2, 2, 3, z2_2az_2, [2, 2]),
        ("bf_p2f2_z2+2az+2_n4", 2, 2, 4, z2_2az_2, [1, 1, 2, 2]),
        ("bf_p2f2_z2+2az+2a_n4", 2, 2, 4, z2_2az_2a, [1, 1, 2, 2]),
        ("bf_p2f2_z2+2az+2a_n5", 2, 2, 5, z2_2az_2a, [1, 1, 1, 2, 3]),
        ("bf_p3f2_z2+3az+3_n3", 3, 2, 3, z2_3az_3, [1, 1, 1, 1]),
        ("bf_p3f2_z2+3az+3a_n4", 3, 2, 4, z2_3az_3a, [1, 1, 2, 2]),
        ("bf_p2f3_z+2_n3", 2, 3, 3, z_2, [1, 1, 2, 2]),
        # hand-computed example (brute force [1,1,2]; without sigma [1,1,1,1])
        ("bf_p2f2_z+2_n3", 2, 2, 3, z_2, [1, 1, 2]),
    ]
    for cid, p, f, n, E, h1 in bf:
        add(cid, p, f, n, 1, E, ["brute-force-i1"],
            refs=[dict(kind="brute-force", source=BRUTE_FORCE_R12A, h1=h1, h2=[])])

    # (b) Hesselholt-Madsen, n <= e; also checked against the formula
    hm = [
        ("hm_p2f2_z2+2az+2_n2_i2", 2, 2, 2, 2, z2_2az_2, [1, 1, 1, 1]),
        ("hm_p2f2_z2+2az+2_n2_i3", 2, 2, 2, 3, z2_2az_2, [1] * 6),
        ("hm_p2f2_z3+2az+2_n3_i2", 2, 2, 3, 2, z3_2az_2, [1, 1, 3, 3]),
        ("hm_p2f2_z3+2az+2_n3_i3", 2, 2, 3, 3, z3_2az_2, [1, 1, 1, 1, 4, 4]),
        ("hm_p3f2_z2+3az+3_n2_i2", 3, 2, 2, 2, z2_3az_3, [2, 2]),
        ("hm_p3f2_z2+3az+3_n2_i3", 3, 2, 2, 3, z2_3az_3, [1, 1, 2, 2]),
    ]
    for cid, p, f, n, i, E, h1 in hm:
        add(cid, p, f, n, i, E, ["hesselholt-madsen"],
            refs=[dict(kind="paper-table", source=HM_R12B, h1=h1, h2=[])])

    # (c) invariance families
    def family(fid, p, f, n, i, members, h1, h2, note, source=INV_R12C):
        ids = []
        for mem in members:
            suffix, E, derived = mem[:3]
            sigma_of = mem[3] if len(mem) > 3 else None
            cid = "inv_%s_%s" % (fid, suffix)
            add(cid, p, f, n, i, E, ["invariance"],
                refs=[dict(kind="sigma-patched-reference", source=source, h1=h1, h2=h2)] if h1 is not None else [],
                derived=derived, sigma_of=sigma_of)
            ids.append(cid)
        families.append(dict(id=fid, p=p, f=f, n=n, i=i, members=ids, note=note))

    g22 = gr[(2, 2)]
    family("p2f2_n4_i2", 2, 2, 4, 2,
           [("E", z2_2az_2, None),
            ("Esigma", g22.poly_sigma(z2_2az_2), "E^sigma of z^2+2az+2 (sigma(a)=a^2, exact since a is Teichmueller)",
             z2_2az_2),
            ("E(az)", g22.poly_scale(z2_2az_2, [0, 1]), "E(a*z) of z^2+2az+2")],
           [1, 1, 2, 2, 3, 4], [1], "E <-> E^sigma <-> E(az)")
    family("p2f2_n2_i2", 2, 2, 2, 2,
           [("z+2", z_2, None), ("az+2", g22.poly_scale(z_2, [0, 1]), "E(a*z) of z+2")],
           [2, 3], [1], "E(z) <-> E(az)")
    g32 = gr[(3, 2)]
    family("p3f2_n2_i2", 3, 2, 2, 2,
           [("z+3", z_3, None), ("az+3", g32.poly_scale(z_3, [0, 1]), "E(a*z) of z+3")],
           [1, 1, 2], [], "E(z) <-> E(az)")
    family("p3f2_n2_i3", 3, 2, 2, 3,
           [("z+3", z_3, None), ("az+3", g32.poly_scale(z_3, [0, 1]), "E(a*z) of z+3")],
           [3, 4], [1], "E(z) <-> E(az)")
    g23 = gr[(2, 3)]
    family("p2f3_n2_i2", 2, 3, 2, 2,
           [("z+2", z_2, None), ("az+2", g23.poly_scale(z_2, [0, 1]), "E(a*z) of z+2"),
            ("(a^2+1)z+2", g23.poly_scale(z_2, [1, 0, 1]), "E((a^2+1)*z) of z+2")],
           [2, 2, 3], [1], "E(z) <-> E(az) <-> E((a^2+1)z)")
    # extra (additional case): input contract E <-> E + p^{ceil(n/e)} h, n=3, e=2
    family("p2f2_n3_i2_perturb", 2, 2, 3, 2,
           [("z2+2", z2_2, None),
            ("z2+4z+2", g22.poly_add_multiple(z2_2, 1, [4]), "z^2+2 + 4z (= E + p^{ceil(n/e)} z)")],
           None, None, "E <-> E + p^{ceil(n/e)} h (extra family, no external reference)")

    # (d) Angeltveit: W_{i(n-1)}(F_q) for i <= p-2
    for n, i in [(2, 2), (2, 3), (3, 2)]:
        m = i * (n - 1)
        add("ang_p5f2_z+5_n%d_i%d" % (n, i), 5, 2, n, i, z_5, ["angeltveit"],
            refs=[dict(kind="formula", source=ANG_R12D + " (H1 = W_{i(n-1)}(F_25) = (Z/5^%d)^2)" % m,
                       h1=[m, m], h2=[])])
    return cases, families


# Negative controls (reference runs without sigma); None = no published value.
NEGATIVE_CONTROLS = [
    ("bf_p2f2_z+2_n3", {"h1": [1, 1, 1, 1], "source": "hand computation without sigma"}),
    ("inv_p2f2_n4_i2_E", {"h1": [1, 1, 2, 2, 4, 4], "h2": [1, 1], "source": "reference run without sigma"}),
    ("inv_p2f2_n4_i2_Esigma", {"h1": [1, 1, 2, 2, 4, 4], "h2": [1, 1], "source": "reference run without sigma"}),
    ("inv_p2f2_n4_i2_E(az)", {"error": "negative valuation", "source": "reference run without sigma"}),
    ("inv_p2f2_n2_i2_z+2", {"h1": [3, 3], "h2": [1, 1], "source": "reference run without sigma"}),
    ("inv_p2f2_n2_i2_az+2", {"h1": [2, 2], "h2": [], "source": "reference run without sigma"}),
    ("inv_p3f2_n2_i2_z+3", {"h1": [1, 1, 1, 1], "source": "reference run without sigma"}),
    ("inv_p3f2_n2_i2_az+3", {"h1": [2, 2], "source": "reference run without sigma"}),
    ("inv_p3f2_n2_i3_z+3", {"h1": [4, 4], "h2": [1, 1], "source": "reference run without sigma"}),
    ("inv_p3f2_n2_i3_az+3", {"h1": [3, 3], "h2": [], "source": "reference run without sigma"}),
]


# ---------------------------------------------------------------------------
# Formulas used by self-checks
# ---------------------------------------------------------------------------

def ilog(x, p):
    """floor(log_p x) for x >= 1, exact."""
    k = 0
    while p ** (k + 1) <= x:
        k += 1
    return k


def hm_formula(p, f, n, i):
    """H^1 exponents of K_{2i-1}(F_q[z]/z^n, (z)) = W_{ni}(F_q)/V_n W_i(F_q) (big Witt vectors),
    via the p-typical decomposition; H^2 = 0."""
    s = valuation(n, p)
    n1 = n // p ** s
    L = n * i
    out = []
    for d in range(1, L + 1):
        if d % p == 0:
            continue
        h = ilog(L // d, p) + 1  # #{j >= 0 : d p^j <= L}
        if d % n1 == 0 and d // n1 <= i:
            e = s
        else:
            e = h
        if e > 0:
            out += [e] * f
    return sorted(out)


def working_precision(G, p, n, i):
    L = n * i
    T = int(G["nygaard_exponent"](p, i, n))
    Mg = ilog(L - 1, p)
    loss_quot = sum(n * valuation(factorial(q), p) for q in range(p, i))
    loss_nyg = n * binomial(i, 2)
    N_apriori = 2 * T + 1 + (Mg + 1) + int(loss_quot) + int(loss_nyg)
    return T, Mg, N_apriori


# ---------------------------------------------------------------------------
# Z_p expansion and cohomology
# ---------------------------------------------------------------------------

def wcoords(w, f, R):
    """Coordinates (in R = Zp) of a Zq element w in the basis 1, a, ..., a^{f-1}."""
    pr = w.precision_absolute()
    lst = w.polynomial().list()
    out = []
    for d in range(f):
        if d < len(lst):
            out.append(R(lst[d]).add_bigoh(pr))
        else:
            out.append(R(0).add_bigoh(pr))
    return out


def mult_matrix(w, a, f, R):
    """f x f matrix of multiplication by w in the basis 1, a, ..., a^{f-1} (column j = w*a^j)."""
    M = Matrix(R, f, f)
    for j in range(f):
        cj = wcoords(w * a ** j, f, R)
        for d in range(f):
            M[d, j] = cj[d]
    return M


def sigma_matrix(a, f, R):
    S = Matrix(R, f, f)
    s = a.frobenius()
    for j in range(f):
        cj = wcoords(s ** j, f, R)
        for d in range(f):
            S[d, j] = cj[d]
    return S


def expand_linear(Mw, a, f, R):
    r, c = Mw.nrows(), Mw.ncols()
    out = Matrix(R, f * r, f * c)
    for x in range(r):
        for y in range(c):
            out.set_block(f * x, f * y, mult_matrix(Mw[x, y], a, f, R))
    return out


def expand_semilinear(Mw, a, f, R, S):
    c = Mw.ncols()
    IS = block_matrix(R, [[S if x == y else Matrix(R, f, f) for y in range(c)] for x in range(c)], subdivide=False)
    return expand_linear(Mw, a, f, R) * IS


def matrix_min_prec(M, cap):
    return min([cap] + [x.precision_absolute() for x in M.list()])


def zz_certificate(M, p, K):
    """Invariant factors of Z^r / (im lift(M) + p^K Z^r); returns sorted p-valuations."""
    r = M.nrows()
    MZ = Matrix(ZZ, r, M.ncols(), [ZZ(x.lift()) for x in M.list()])
    big = MZ.augment(p ** K * identity_matrix(ZZ, r))
    eds = big.elementary_divisors()
    vals = []
    for e in eds:
        e = ZZ(e)
        if e == 0:
            vals.append(K)
        else:
            vals.append(min(K, valuation(e, p)))
    return sorted(vals)


def padic_ed_exponents(M, p):
    out, zeros = [], 0
    for e in M.elementary_divisors():
        if e == 0:
            zeros += 1
        else:
            v = e.valuation()
            if v > 0:
                out.append(int(v))
    return sorted(out), zeros


def int_coords(w, f):
    """Integer lifts of the coordinates of a Zq element (basis 1, a, ..., a^{f-1})."""
    lst = w.polynomial().list()
    return [ZZ(lst[d].lift()) if d < len(lst) else ZZ(0) for d in range(f)]


def wval(w):
    """p-adic valuation of a Zq element, or None for an (inexact) zero."""
    return None if w.is_zero() else int(w.valuation())


def structural_checks(mats, p, n, i):
    """Plan 'Matrix invariants' at the W level (F-weight order, 0-based indices as in INDEX_CONVENTIONS)."""
    L = n * i
    out = {}
    for k in ["can0", "can1", "red", "redN", "nablaOK"]:
        M = mats[k]
        out["%s_lower_triangular" % k] = all(M[r, c].is_zero() for r in range(M.nrows())
                                            for c in range(r + 1, M.ncols()))
    M = mats["phi0"]
    out["phi0_support_rows_weight_ge_p_times_col_weight"] = all(
        M[r, c].is_zero() for r in range(M.nrows()) for c in range(M.ncols()) if r + 1 < p * (c + 1))
    out["can0_diag_val_eq_i-floor(j/n)"] = all(wval(mats["can0"][j - 1, j - 1]) == i - j // n for j in range(1, L))
    out["can1_diag_val_eq_i-1-floor(m/n)"] = all(wval(mats["can1"][m, m]) == i - 1 - m // n for m in range(0, L - 1))
    out["red_diag_val_eq_vp(floor(j/n)!)"] = all(
        wval(mats["red"][j - 1, j - 1]) == valuation(factorial(j // n), p) for j in range(1, L))
    out["redN_diag_val_eq_vp(floor(m/n)!)"] = all(
        wval(mats["redN"][m, m]) == valuation(factorial(m // n), p) for m in range(0, L - 1))
    out["nablaOK_diag_val_eq_vp(k)"] = all(wval(mats["nablaOK"][c, c]) == valuation(c + 1, p) for c in range(0, L - 1))
    return out


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def zp_str(x, prec, p):
    """Decimal string of a representative in [0, p^prec)."""
    return str(ZZ(x.lift()) % (p ** prec)) if prec > 0 else "0"


def welt_json(w, f, R, p):
    pr = w.precision_absolute()
    cs = wcoords(w, f, R)
    return [zp_str(c, pr, p) for c in cs], int(pr)


def wmatrix_json(Mw, f, R, p, N, convention):
    ents, precs = [], []
    for x in range(Mw.nrows()):
        row, prow = [], []
        for y in range(Mw.ncols()):
            e, pr = welt_json(Mw[x, y], f, R, p)
            row.append(e)
            prow.append(pr)
        ents.append(row)
        precs.append(prow)
    return {"rows": int(Mw.nrows()), "cols": int(Mw.ncols()), "index_convention": convention,
            "entries": ents, "prec": precs}


INDEX_CONVENTIONS = {
    "can0": ("Step 9 input: W-linear can: N^{>=i} -> Prism of R = O_K/pi^n. Rows and columns are the "
             "reduced W-basis of F-weights 1..L-1 (L = n*i): entry [r][c] (0-based) is the coefficient, "
             "on the basis element of F-weight r+1 (z^{(r+1) mod n} * prod_u f_u^{digit_u(floor((r+1)/n))}), "
             "of NF_R(E^{i-floor((c+1)/n)} * z^{(c+1) mod n} * prod_u f_u^{digit_u(floor((c+1)/n))}) "
             "(Sage L629-646). Z_p expansion: M(can0)."),
    "phi0": ("Step 9 input: sigma-SEMILINEAR phi_i: N^{>=i} -> Prism; indices as can0; column c is "
             "NF_R(z^{p*((c+1) mod n)} * prod_u phi_{p^u}(f_u)^{digit_u(floor((c+1)/n))}) (Sage L654-671, "
             "with sigma-patched Frobenius). Z_p expansion: M(phi0)*(I kron S)."),
    "can1": ("Step 8 input: W-linear can: N^{>=i-1} -> Prism (nabla side). Rows and columns are F-weights "
             "0..L-2: entry [r][c] is the coefficient, on the basis element of F-weight r, of "
             "NF_R(E^{i-1-floor(c/n)} * z^{c mod n} * prod_u f_u^{digit_u(floor(c/n))}) (Sage L676-693). "
             "Z_p expansion: M(can1)."),
    "red": ("Step 6 red: Prism_OK -> Prism_R on F-weights 1..L-1: entry [r][c] is the coefficient of the "
            "basis element of F-weight r+1 in NF_R(z^{c+1}) (Sage L706-717, OKtoOKmodpi0). Z_p expansion: M(red)."),
    "redN": ("Step 6 red^nabla on F-weights 0..L-2: entry [r][c] is the coefficient of the basis element of "
             "F-weight r in NF_R(z^c) (Sage L719-730, OKtoOKmodpi1). Z_p expansion: M(redN)."),
    "nablaOK": ("Step 5 nabla on Prism_OK in Sage's sign convention g0 = z0 - z1 (eta_R(z) = z - g0): entry "
                "[r][c] (r, c in 0..L-2) is the coefficient of z^r in the g0-coefficient (all other g_u "
                "exponents 0) of NF_OK(-(z - g0)^{c+1} * v^i), v = prod_{r>=0} phi^r(u) (Sage L587-600). "
                "For odd p this equals the paper-convention matrix; for p = 2 it differs by units (compare "
                "cohomology only). Z_p expansion: M(nablaOK)."),
}


# ---------------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------------

def _frob_sigma_less(c):
    """Negative control only: the upstream f = 1 Frobenius c(z) -> c(z^p) (no sigma)."""
    return c.V(res_G["p"])


def run_sage_case(G, case, N, sigma=True):
    p, f, n, i = case["p"], case["f"], case["n"], case["i"]
    L = n * i
    q = p ** f
    W = Zq(q, N, names="a", type="capped-abs", print_mode="series", show_prec=False)
    a = W.gen()
    A = PowerSeriesRing(W, "z", L)
    z = A.gen()
    # The KZPN code reads these globals (p, i, n inside prismatic_envelope_f/g; W, A, z everywhere).
    G.update(dict(p=p, i=i, n=n, W=W, A=A, z=z, a=a, Fprec=L, k=n))
    Eraw = A(sum(sum(W(c) * a ** d for d, c in enumerate(coef)) * z ** kk for kk, coef in enumerate(case["E"])))
    # KZPN's eisenstein_normalization returns a series over Frac(W); over Zq the coercion of that
    # series into B = A[f_0, ...] fails in Sage 10.3, so the (integral) result is converted back to A.
    E = A(G["eisenstein_normalization"](p, Eraw))
    G["E"] = E
    t0 = time.time()
    frob = G["FROB"]
    if not sigma:
        G["FROB"] = _frob_sigma_less
    try:
        can0, phi0, can1, red, redN, nablaOK = G["syntomic_matrices"](p, i, n, E, N, L)
    finally:
        G["FROB"] = frob
    t_sage = time.time() - t0
    return dict(W=W, a=a, A=A, Eraw=Eraw, E=E, mats=dict(can0=can0, phi0=phi0, can1=can1, red=red,
                                                          redN=redN, nablaOK=nablaOK), t_sage=t_sage)


def cohomology_from_W(res, case, N, sigma=True):
    p, f = case["p"], case["f"]
    W, a = res["W"], res["a"]
    R = W.base_ring()
    S = sigma_matrix(a, f, R) if sigma else identity_matrix(R, f)
    mz = {k: expand_linear(v, a, f, R) for k, v in res["mats"].items() if k != "phi0"}
    phi0Z = expand_semilinear(res["mats"]["phi0"], a, f, R, S)
    sq_bottom = res_G["square_complete_bottom"]
    sq_top = res_G["square_complete_top"]
    sq_right = res_G["square_complete_right"]
    # Step 7: nabla_R = redN * nablaOK * red^{-1}
    _, _, _, nablaP = sq_bottom(mz["red"], mz["redN"], mz["nablaOK"])
    # Step 9: can - phi
    syn0 = mz["can0"] - phi0Z
    # Step 8: N nabla = can1^{-1} * nabla_R * can0
    _, _, nablaN, _ = sq_top(mz["can0"], mz["can1"], nablaP)
    # Step 10: can - phi^nabla = nabla_R * (can - phi) * N nabla^{-1}
    _, syn1, _, _ = sq_right(syn0, nablaN, nablaP)
    d0 = block_matrix(R, [[syn0], [nablaN]], subdivide=False)
    d1 = block_matrix(R, [[nablaP, -syn1]], subdivide=False)
    M0 = matrix_min_prec(d0, N)
    M1 = matrix_min_prec(d1, N)
    h1_sage, z0 = padic_ed_exponents(d0, p)
    h2_sage, z1 = padic_ed_exponents(d1, p)
    m = d0.ncols()
    v0 = zz_certificate(d0, p, M0)
    v1 = zz_certificate(d1, p, M1)
    # d0: 2m x m -> m invariant factors below p^{M0} (full rank), the other m equal p^{M0}
    cert0_ok = (len([v for v in v0 if v < M0]) == m)
    cert1_ok = (len(v1) == d1.nrows() and all(v < M1 for v in v1))
    h1_zz = sorted([v for v in v0 if 0 < v < M0])
    h2_zz = sorted([v for v in v1 if 0 < v < M1])
    # d1 * d0 = 0 bookkeeping
    dd = d1 * d0
    dd_val = min([N] + [x.valuation() for x in dd.list() if not x.is_zero()])
    return dict(S=S, M0=int(M0), M1=int(M1), h1=h1_sage, h2=h2_sage, zero_ed=(z0, z1),
                h1_zz=h1_zz, h2_zz=h2_zz, cert0_ok=cert0_ok, cert1_ok=cert1_ok, rank=int(m),
                d1d0_min_val=int(dd_val))


res_G = None  # the KZPN namespace (set in main)


def mats_agree(m1, m2, f, p):
    """Entrywise agreement of two sets of W-matrices (possibly over different Zq caps) modulo the
    smaller absolute precision of each entry pair."""
    for k in m1:
        A1, A2 = m1[k], m2[k]
        if A1.dimensions() != A2.dimensions():
            return False, k
        for x, y in zip(A1.list(), A2.list()):
            pr = min(x.precision_absolute(), y.precision_absolute())
            mod = ZZ(p) ** pr
            if any((u - v) % mod != 0 for u, v in zip(int_coords(x, f), int_coords(y, f))):
                return False, k
    return True, None


def case_json(case, res, coh, N, T, Mg, N_apriori, h_tilde, extra):
    p, f = case["p"], case["f"]
    W, a, A = res["W"], res["a"], res["A"]
    R = W.base_ring()
    E_exact = case["E"]
    E_json = [[str(ZZ(c) % p ** N) for c in coef] for coef in E_exact]
    En = A(res["E"])
    En_list = En.list()
    e_deg = len(E_exact) - 1
    En_json, En_prec = [], []
    for kk in range(e_deg + 1):
        w = En_list[kk] if kk < len(En_list) else W(0)
        ej, pr = welt_json(w, f, R, p)
        En_json.append(ej)
        En_prec.append(pr)
    S = coh["S"]
    out = {
        "id": case["id"],
        "tags": case["tags"],
        "p": p, "f": f, "n": case["n"], "i": case["i"], "L": case["n"] * case["i"], "e": e_deg,
        "h_tilde": [str(c) for c in h_tilde],
        "E_display": fmt_poly(E_exact),
        "E_derivation": case.get("derived"),
        "E": E_json,
        "E_prec": int(N),
        "E_exact_signed": [[str(ZZ(c)) for c in coef] for coef in E_exact],
        "E_normalized": En_json,
        "E_normalized_prec": En_prec,
        "N": int(N),
        "N_apriori_plan": int(N_apriori),
        "target_T": int(T),
        "M_g": int(Mg),
        "sigma_matrix_S": {"rows": f, "cols": f,
                           "index_convention": "S[d][j] = d-th coordinate of sigma(a^j) in the basis 1, a, ..., a^{f-1}",
                           "entries": [[zp_str(S[d, j], S[d, j].precision_absolute(), p) for j in range(f)] for d in range(f)],
                           "prec": [[int(S[d, j].precision_absolute()) for j in range(f)] for d in range(f)]},
    }
    out["matrices"] = {k: wmatrix_json(res["mats"][k], f, R, p, N, INDEX_CONVENTIONS[k])
                       for k in ["can0", "phi0", "can1", "red", "redN", "nablaOK"]}
    out["h1"] = [int(v) for v in coh["h1"]]
    out["h2"] = [int(v) for v in coh["h2"]]
    out["precision_of_d0_d1"] = {"M0": coh["M0"], "M1": coh["M1"]}
    out["references"] = case["refs"]
    out.update(extra)
    return out


def plain(obj):
    """Recursively convert Sage integers (only small metadata: sizes, exponents, flags) to Python ints.
    All p-adic numbers are already decimal strings at this point."""
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, Integer):
        v = int(obj)
        if abs(v) >= 2 ** 53:
            raise ValueError("large integer would become a JSON number")
        return v
    if isinstance(obj, (bool, str, float)) or obj is None:
        return obj
    if isinstance(obj, int):
        if abs(obj) >= 2 ** 53:
            raise ValueError("large integer would become a JSON number")
        return obj
    raise TypeError("unexpected object in JSON document: %r" % (type(obj),))


# ---------------------------------------------------------------------------
# Julia/Nemo h_tilde check
# ---------------------------------------------------------------------------

def nemo_defining_polynomials(pfs):
    code = ("using Nemo, Hecke; for (p,f) in [%s]; K,_ = qadic_field(p,f; precision=10); "
            "h = defining_polynomial(K); println(p, \" \", f, \" \", join([string(lift(ZZ, coeff(h,j))) "
            "for j in 0:degree(h)], \" \")); end") % ", ".join("(%d,%d)" % pf for pf in pfs)
    cmd = ["julia", "--project=%s" % REPO_ROOT, "--startup-file=no", "-e", code]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except Exception as ex:  # pragma: no cover
        return None, "julia failed to run: %r" % (ex,)
    if out.returncode != 0:
        return None, "julia exited %d: %s" % (out.returncode, out.stderr[-2000:])
    res = {}
    for line in out.stdout.strip().splitlines():
        parts = line.split()
        res[(int(parts[0]), int(parts[1]))] = [int(t) for t in parts[2:]]
    return res, "%s; using Nemo, Hecke; defining_polynomial(qadic_field(p, f; precision=10))" % (
        subprocess.run(["julia", "--version"], capture_output=True, text=True).stdout.strip())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global res_G
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True, help="generation date recorded in provenance (string)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--only", nargs="*", default=None, help="run only these case ids")
    ap.add_argument("--margin", type=int, default=None,
                    help="extra digits over the plan's :apriori N (default: 2*L + 10)")
    ap.add_argument("--recheck-extra", type=int, default=8,
                    help="extra digits for the stability rerun (0 disables it)")
    ap.add_argument("--no-julia-check", action="store_true")
    args = ap.parse_args()
    if args.only and os.path.abspath(args.out) == os.path.abspath(DEFAULT_OUT):
        raise SystemExit("--only writes a partial fixture; pass --out PATH so the committed "
                         "test/fixtures/sage_fq.json is not overwritten")

    kzpn_dir = os.path.abspath(os.environ.get("KZPN_DIR", DEFAULT_KZPN))
    shas = {fn: sha256_file(os.path.join(kzpn_dir, fn)) for fn in KZPN_FILES}
    G, applied = load_kzpn(kzpn_dir)
    res_G = G

    cases, families = build_cases()
    if args.only:
        cases = [c for c in cases if c["id"] in set(args.only)]

    # ---- h_tilde verification (Sage Zq modulus == Conway lift == Nemo/Hecke) ----
    from sage.all import conway_polynomial
    pfs = sorted(set((c["p"], c["f"]) for c in cases))
    htilde_checks = {}
    for pf in pfs:
        p, f = pf
        Wt = Zq(p ** f, 5, names="a", type="capped-abs")
        sage_mod = [ZZ(c.lift()) if hasattr(c, "lift") else ZZ(c) for c in Wt.modulus().list()]
        conway = [ZZ(c) for c in conway_polynomial(p, f).change_ring(ZZ).list()]
        htilde_checks[pf] = dict(sage_Zq_modulus=[str(c) for c in sage_mod], sage_conway=[str(c) for c in conway],
                                 script_table=[str(c) for c in CONWAY[pf]])
        if not (sage_mod == conway == CONWAY[pf]):
            raise RuntimeError("h_tilde mismatch for %r: %r" % (pf, htilde_checks[pf]))
    nemo_note = "skipped (--no-julia-check)"
    if not args.no_julia_check:
        nemo, nemo_note = nemo_defining_polynomials(pfs)
        for pf in pfs:
            if nemo is None:
                htilde_checks[pf]["nemo"] = None
                htilde_checks[pf]["nemo_match"] = None
            else:
                htilde_checks[pf]["nemo"] = [str(c) for c in nemo[pf]] if pf in nemo else None
                htilde_checks[pf]["nemo_match"] = (nemo.get(pf) == CONWAY[pf])
                if nemo.get(pf) != CONWAY[pf]:
                    raise RuntimeError("Nemo defining polynomial differs for %r" % (pf,))

    results, skipped, failures = [], [], []
    by_id = {}
    for case in cases:
        p, f, n, i = case["p"], case["f"], case["n"], case["i"]
        L = n * i
        T, Mg, N_apriori = working_precision(G, p, n, i)
        margin = args.margin if args.margin is not None else 2 * L + 10
        N = N_apriori + margin
        t0 = time.time()
        try:
            alarm(CASE_TIMEOUT_S)
            res = run_sage_case(G, case, N)
            coh = cohomology_from_W(res, case, N)
            cancel_alarm()
        except AlarmInterrupt:
            skipped.append(dict(id=case["id"], reason="timeout > %d s at N=%d" % (CASE_TIMEOUT_S, N)))
            print("[skip] %s: timeout" % case["id"], flush=True)
            continue
        except Exception as ex:
            cancel_alarm()
            failures.append(dict(id=case["id"], error=repr(ex), traceback=traceback.format_exc()[-3000:]))
            print("[FAIL] %s: %r" % (case["id"], ex), flush=True)
            continue
        t_case = time.time() - t0
        checks = {}
        # certificate: integer Smith form agrees with Sage's p-adic elementary divisors
        checks["zz_certificate_d0_full_rank"] = bool(coh["cert0_ok"])
        checks["zz_certificate_d1_full_rank"] = bool(coh["cert1_ok"])
        checks["zz_matches_padic_h1"] = (coh["h1_zz"] == coh["h1"])
        checks["zz_matches_padic_h2"] = (coh["h2_zz"] == coh["h2"])
        checks["no_zero_elementary_divisors"] = (coh["zero_ed"] == (0, 0))
        checks["exponents_below_precision"] = (all(v < coh["M0"] for v in coh["h1"]) and
                                               all(v < coh["M1"] for v in coh["h2"]))
        checks["order_bookkeeping_sumH1_minus_sumH2_eq_f_i_(n-1)"] = (
            sum(coh["h1"]) - sum(coh["h2"]) == f * i * (n - 1))
        # bookkeeping only: syn1 is defined by the square completion, so d1*d0 = 0 up to precision
        checks["bookkeeping_d1_d0_min_valuation"] = coh["d1d0_min_val"]
        checks.update(structural_checks(res["mats"], p, n, i))
        if case.get("sigma_of") is not None:
            # the exact integer coordinates of E^sigma agree with Frobenius computed in Zq
            W, a = res["W"], res["a"]
            ok = True
            for cs, cb in zip(case["E"], case["sigma_of"]):
                wb = sum(W(c) * a ** d for d, c in enumerate(cb)).frobenius()
                ws = sum(W(c) * a ** d for d, c in enumerate(cs))
                ok = ok and (wb - ws).is_zero()
            checks["E_sigma_coords_match_Zq_frobenius"] = bool(ok)
        if "hesselholt-madsen" in case["tags"]:
            hmf = hm_formula(p, f, n, i)
            checks["hm_formula_h1"] = hmf
            checks["hm_formula_match"] = (hmf == coh["h1"] and coh["h2"] == [])
        # stability rerun at higher precision
        if args.recheck_extra > 0:
            try:
                alarm(CASE_TIMEOUT_S)
                res2 = run_sage_case(G, case, N + args.recheck_extra)
                coh2 = cohomology_from_W(res2, case, N + args.recheck_extra)
                cancel_alarm()
                agree, where = mats_agree(res["mats"], res2["mats"], f, p)
                checks["rerun_N"] = int(N + args.recheck_extra)
                checks["rerun_same_h1_h2"] = (coh2["h1"] == coh["h1"] and coh2["h2"] == coh["h2"])
                checks["rerun_W_matrices_agree_mod_min_prec"] = bool(agree)
                if not agree:
                    checks["rerun_first_disagreeing_matrix"] = where
            except AlarmInterrupt:
                checks["rerun_N"] = "timeout"
            except Exception as ex:
                cancel_alarm()
                checks["rerun_ok"] = False
                checks["rerun_error"] = repr(ex)
        for ref in case["refs"]:
            key = "ref_%s_match" % ref["kind"]
            checks[key] = (ref["h1"] == coh["h1"] and ref["h2"] == coh["h2"])
        extra = {"timing_s": {"sage_W_matrices": round(res["t_sage"], 3), "total": round(t_case, 3)},
                 "checks": checks}
        cj = case_json(case, res, coh, N, T, Mg, N_apriori, CONWAY[(p, f)], extra)
        results.append(cj)
        by_id[case["id"]] = cj
        bad = [k for k, v in checks.items() if v is False]
        print("[ok] %-34s N=%3d M0=%3d M1=%3d h1=%s h2=%s t=%.2fs%s" % (
            case["id"], N, coh["M0"], coh["M1"], coh["h1"], coh["h2"], t_case,
            ("  CHECK FAILURES: %s" % bad) if bad else ""), flush=True)

    # ---- negative controls: the upstream sigma-less code (V(p) Frobenius, S = I) ----
    controls = []
    for cid, ref in NEGATIVE_CONTROLS:
        case = next((c for c in cases if c["id"] == cid), None)
        if case is None or cid not in by_id:
            continue
        p, f, n, i = case["p"], case["f"], case["n"], case["i"]
        N = by_id[cid]["N"]
        row = dict(id=cid, E=by_id[cid]["E_display"], n=n, i=i, sigma_h1=by_id[cid]["h1"], sigma_h2=by_id[cid]["h2"],
                   sigma_less_reference=ref)
        try:
            alarm(CASE_TIMEOUT_S)
            resc = run_sage_case(G, case, N, sigma=False)
            cohc = cohomology_from_W(resc, case, N, sigma=False)
            cancel_alarm()
            row.update(sigma_less_h1=cohc["h1"], sigma_less_h2=cohc["h2"], error=None)
            row["differs_from_sigma_result"] = (cohc["h1"], cohc["h2"]) != (by_id[cid]["h1"], by_id[cid]["h2"])
        except AlarmInterrupt:
            row.update(error="timeout", differs_from_sigma_result=None)
        except Exception as ex:
            cancel_alarm()
            row.update(sigma_less_h1=None, sigma_less_h2=None, error=repr(ex)[:300], differs_from_sigma_result=True)
        if ref is not None:
            if "error" in ref:
                row["matches_sigma_less_reference"] = (row["error"] is not None and ref["error"] in row["error"])
            else:
                row["matches_sigma_less_reference"] = (row.get("sigma_less_h1") == ref["h1"] and
                                                    ("h2" not in ref or row.get("sigma_less_h2") == ref["h2"]))
        controls.append(row)
        print("[control] %-30s sigma-less h1=%s h2=%s err=%s" % (cid, row.get("sigma_less_h1"), row.get("sigma_less_h2"),
                                                              row["error"]), flush=True)

    # ---- family / cross-case self-checks ----
    self_checks = {"invariance_families": [], "brute_force_i1": [], "hesselholt_madsen": [],
                   "angeltveit": [], "negative_controls_sigma_less": controls, "per_case_failures": []}
    for fam in families:
        mem = [by_id[m] for m in fam["members"] if m in by_id]
        if len(mem) < 2:
            continue
        vals = [(m["h1"], m["h2"]) for m in mem]
        entry = dict(family=fam["id"], note=fam["note"], p=fam["p"], f=fam["f"], n=fam["n"], i=fam["i"],
                     members=[dict(id=m["id"], E=m["E_display"], h1=m["h1"], h2=m["h2"]) for m in mem],
                     identical=all(v == vals[0] for v in vals))
        refs = [r for m in mem for r in m["references"]]
        if refs:
            entry["matches_reference"] = all(r["h1"] == vals[0][0] and r["h2"] == vals[0][1] for r in refs)
        self_checks["invariance_families"].append(entry)
    for c in results:
        for ref in c["references"]:
            row = dict(id=c["id"], E=c["E_display"], n=c["n"], i=c["i"], computed_h1=c["h1"], computed_h2=c["h2"],
                       reference_h1=ref["h1"], reference_h2=ref["h2"], source=ref["source"],
                       match=(ref["h1"] == c["h1"] and ref["h2"] == c["h2"]))
            if "brute-force-i1" in c["tags"]:
                self_checks["brute_force_i1"].append(row)
            elif "hesselholt-madsen" in c["tags"]:
                row["hm_formula_h1"] = c["checks"].get("hm_formula_h1")
                row["hm_formula_match"] = c["checks"].get("hm_formula_match")
                self_checks["hesselholt_madsen"].append(row)
            elif "angeltveit" in c["tags"]:
                self_checks["angeltveit"].append(row)
        bad = [k for k, v in c["checks"].items() if v is False]
        if bad:
            self_checks["per_case_failures"].append(dict(id=c["id"], failed=bad))
    all_ok = (not failures and not self_checks["per_case_failures"]
              and all(e["identical"] and e.get("matches_reference", True) for e in self_checks["invariance_families"])
              and all(r["match"] for k in ("brute_force_i1", "hesselholt_madsen", "angeltveit") for r in self_checks[k])
              and all(r["differs_from_sigma_result"] and r.get("matches_sigma_less_reference", True) for r in controls))
    self_checks["all_passed"] = bool(all_ok)

    doc = {
        "provenance": {
            "generator": "scripts/sage/make_fixtures_fq.py",
            "sage_version": _sage_version.version,
            "sage_version_banner": version(),
            "generation_date": args.date,
            "kzpn_dir_basename": os.path.basename(kzpn_dir.rstrip("/")),
            "kzpn_sha256": shas,
            "patch_set_id": PATCH_SET_ID,
            "patches": applied,
            "monkeypatches": MONKEYPATCHES,
            "harness_notes": HARNESS_NOTES,
            "sign_convention": ("nablaOK is expressed in the generators g_u of the Sage implementation: for odd "
                                "p they are the negatives of the generators of [AKN, Construction 3.28] and "
                                "nablaOK is the matrix of [AKN, Section 7.1, Step 5]; for p = 2 they differ "
                                "by units, so only the cohomology is compared."),
            "trust": "sage-sigma-patched",
            "trust_note": ("Upstream KZPN hard-codes f = 1; these fixtures come from an in-memory sigma-patched "
                           "copy. Lower trust than f = 1 Sage fixtures; validated by "
                           "the self_checks below."),
            "working_precision_rule": ("N = N_apriori_plan + margin, N_apriori_plan = 2T + 1 + (M_g+1) + "
                                       "sum_{s=p}^{i-1} n v_p(s!) + n C(i,2), margin = 2L + 10 unless --margin"),
            "h_tilde_checks": {"%d,%d" % pf: v for pf, v in htilde_checks.items()},
            "nemo_check": nemo_note,
            "timeout_per_case_s": CASE_TIMEOUT_S,
        },
        "conventions": {
            "p_adic_numbers": "decimal strings of representatives in [0, p^prec)",
            "W_element": ("list of f decimal strings: coordinates in the basis 1, a, ..., a^(f-1) of "
                          "Z_p[x]/(h_tilde), a = x the Conway generator; h_tilde is the Conway lift (low -> high)"),
            "E": ("E: exact integer coordinates reduced into [0, p^E_prec) with E_prec = N (the case working "
                  "precision); E_exact_signed: the same exact integers with sign (e.g. E^sigma, E(az) have "
                  "negative coordinates). E_normalized = E/(E(0)/p) as computed by Sage (per-coefficient "
                  "absolute precision in E_normalized_prec)."),
            "polynomials": "lists low -> high degree of W-elements",
            "matrices": ("case[\"matrices\"][name] for name in can0, phi0, can1, red, redN, nablaOK (same layout "
                         "as sage_f1.json): {rows, cols, index_convention, entries (row-major W-elements), prec "
                         "(absolute precision per entry as reported by Sage; exact zeros carry N)}. "
                         "case[\"sigma_matrix_S\"] is a Z_p-matrix: its entries are single decimal strings, not "
                         "W-element lists"),
            "zp_expansion": ("W-linear A -> M(A) (block (r,c) = f x f matrix of multiplication by A[r][c], column "
                             "j = coordinates of A[r][c]*a^j); sigma-semilinear Phi -> M(Phi)*(I kron S); the "
                             "1-based index of a^d * b_m is f*(m-1)+d+1"),
            "square_completions": ("over Zp capped-abs after expansion: nabla_R = M(redN) M(nablaOK) M(red)^-1; "
                                   "can-phi = M(can0) - M(phi0)(I kron S); N nabla = M(can1)^-1 nabla_R M(can0); "
                                   "can-phi^nabla = nabla_R (can-phi) (N nabla)^-1; d0 = [can-phi ; N nabla], "
                                   "d1 = [nabla_R, -(can-phi^nabla)]"),
            "cohomology": ("h1 = torsion of coker d0, h2 = coker d1, as sorted exponent lists (Z/p^a); [] = 0; "
                           "H0 = 0 (Cor 2.14). K_{2i-1}(R; Z_p) = H1, K_{2i-2}(R; Z_p) = H2 for i >= 2."),
            "paper_vs_sage_notation": "paper n = Sage k (power of the uniformizer), i = weight, L = n*i = Sage Fprec",
            "convention_dependence": ("can0, phi0, can1, red, redN are f-side matrices and do not depend on the g-side "
                                      "sign convention; nablaOK depends on the generators g_u (see provenance.sign_convention)"),
        },
        "cases": results,
        "self_checks": self_checks,
        "skipped": skipped,
        "failures": failures,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    doc = plain(doc)
    with open(args.out, "w") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    print("wrote %s: %d cases, %d skipped, %d failed, all self-checks passed: %s" % (
        args.out, len(results), len(skipped), len(failures), all_ok), flush=True)


if __name__ == "__main__":
    main()
