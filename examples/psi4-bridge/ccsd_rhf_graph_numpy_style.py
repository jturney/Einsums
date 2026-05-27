#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Closed-shell hybrid DF-CCSD as a ComputeGraph LOOP (graph-numpy-style).

The CCSD iteration is a graph add_loop: body captured ONCE, re-executed by the
loop executor. Unlike the eager numpy-style version, a captured loop body must
use PRE-ALLOCATED scratch + in-place ops (operator-created temporaries get GC'd
-> pointer reuse -> graph aliasing). So intermediates/residuals are graph-owned
scratch recomputed in place each iteration, amplitudes t1/t2 update in place,
and a convergence predicate reads the energy tensor. Integrals (bridge) + Fock +
denominators are one-time eager setup.

HYBRID DF: the v⁴ particle-ladder integral <ab|ef> = (ae|bf) is the only block
taken from density fitting, <ab|ef> = Σ_Q B^Q_ae B^Q_bf with B = J^{-1/2}(Q|vv)
from psi4's DFTensor. The v⁴ tensor is NEVER formed — the ladder contraction
splits into two o²v³ steps through a DF intermediate. Every other block stays
EXACT via the half-transform bridge. Validate the DF shift vs psi4 conv CCSD.

    PYTHONPATH=/Users/jturney/Code/Einsums/Einsums/build/lib:/Users/jturney/Code/psi4/cmake-build-debug/stage/lib \
        /Users/jturney/miniconda3/envs/einsums-dev/bin/python \
        examples/psi4-bridge/ccsd_rhf_graph_numpy_style.py

"""
import numpy as np
import psi4
import einsums
import einsums._core
import einsums.graph as cg
from einsums import linalg as la

psi4.core.set_output_file("/tmp/p.out", False)
psi4.set_options({"basis": "cc-pvdz", "scf_type": "pk", "freeze_core": "false",
                  "e_convergence": 1e-10, "d_convergence": 1e-10, "cc_type": "conv", "r_convergence": 1e-9})
mol = psi4.geometry("O\nH 1 0.96\nH 1 0.96 2 104.5\nsymmetry c1\n")
e_scf, wfn = psi4.energy("ccsd", return_wfn=True)
ref = psi4.variable("CCSD CORRELATION ENERGY")

nbf, nocc = wfn.nso(), wfn.nalpha(); nv = nbf - nocc
C = np.asarray(wfn.Ca()); eps = np.asarray(wfn.epsilon_a())
eo, ev = eps[:nocc], eps[nocc:]
Co_np = np.ascontiguousarray(C[:, :nocc]); Cv_np = np.ascontiguousarray(C[:, nocc:])
mints = psi4.core.MintsHelper(wfn.basisset())
Co = psi4.core.Matrix.from_array(Co_np); Cv = psi4.core.Matrix.from_array(Cv_np)

# ── DF 3-index for the v⁴ particle-ladder (true DF-CCSD: <ab|ef>=(ae|bf) from B,
#    the v⁴ tensor is NEVER formed). Everything else stays EXACT via the bridge. ─
aux = psi4.core.BasisSet.build(mol, "DF_BASIS_MP2", "", "RIFIT", wfn.basisset().name())
naux = aux.nbf()
dft = psi4.core.DFTensor(wfn.basisset(), aux, wfn.Ca(), nocc, nv)
Bvv = dft.Qvv_einsums()   # B^Q_{ab} = J^{-1/2}(Q|ab), einsums RuntimeTensor (naux, nv, nv)

def to_t(name, a): return einsums.asarray(np.ascontiguousarray(a), name=name)
def zt(name, shape): return einsums.create_zero_tensor(name, list(shape), dtype="float64")

# ── one-time eager setup: integrals from the bridge, Fock, denominators ──────
HT = {("o", "o"): mints.mo_bra_half_transform_einsums(Co, Co),
      ("o", "v"): mints.mo_bra_half_transform_einsums(Co, Cv),
      ("v", "v"): mints.mo_bra_half_transform_einsums(Cv, Cv)}
Ct = {"o": to_t("Co", Co_np), "v": to_t("Cv", Cv_np)}; nsp = {"o": nocc, "v": nv}

def E(spec, A, B, shape, pf=1.0, name="x"):  # eager helper for one-time setup
    out = zt(name, shape); einsums.einsum(spec, out, A, B, c_pf=0.0, ab_pf=pf); return out

def phys(P, Q, R, S, name):
    ht = HT[(P, R)]; nP, nR, nQ, nS = nsp[P], nsp[R], nsp[Q], nsp[S]
    tmp = E("a,b,q,sig <- lam,q ; a,b,lam,sig", Ct[Q], ht, (nP, nR, nQ, nbf), name=name + "_t")
    chem = E("a,b,q,s <- sig,s ; a,b,q,sig", Ct[S], tmp, (nP, nR, nQ, nS), name=name + "_c")
    out = zt(name, (nP, nQ, nR, nS)); einsums.permute("a,q,b,s <- a,b,q,s", out, chem); return out

specs = {"oovv": ("o", "o", "v", "v"), "oooo": ("o", "o", "o", "o"),
         "ovvo": ("o", "v", "v", "o"), "ovov": ("o", "v", "o", "v"), "ooov": ("o", "o", "o", "v"),
         "oovo": ("o", "o", "v", "o"), "ovvv": ("o", "v", "v", "v"), "vvvo": ("v", "v", "v", "o"),
         "ovoo": ("o", "v", "o", "o")}  # no "vvvv": the v⁴ block is DF, never formed
G = {nm: phys(*pqrs, nm) for nm, pqrs in specs.items()}

# diagnostic only: how well does the DF B reconstruct the exact v⁴ block? (both
# tensors formed here are discarded — the iteration never touches them). This Δ is
# the only new approximation the DF swap introduces vs the exact-v⁴ run (1.26e-11).
# Reconstruction is einsums-native; numpy is used only to read the scalar error.
_vvvv_exact = phys("v", "v", "v", "v", "vvvv_diag")
_vvvv_df = E("a,b,e,f <- Q,a,e ; Q,b,f", Bvv, Bvv, (nv, nv, nv, nv), name="vvvv_df_diag")
print(f"DF v⁴ reconstruction error  max|<ab|ef>_exact - <ab|ef>_DF| = "
      f"{np.abs(np.asarray(_vvvv_exact) - np.asarray(_vvvv_df)).max():.3e}")
Fvv = to_t("Fvv", np.diag(ev)); Foo = to_t("Foo", np.diag(eo))
Dia = to_t("Dia", eo[:, None] - ev[None, :])
Dijab = to_t("Dijab", eo[:, None, None, None] + eo[None, :, None, None]
            - ev[None, None, :, None] - ev[None, None, None, :])
oovv_ba = zt("oovv_ba", (nocc, nocc, nv, nv)); einsums.permute("i,j,b,a <- i,j,a,b", oovv_ba, G["oovv"])

NV2 = (nocc, nv); SH2 = (nocc, nocc, nv, nv); VV = (nv, nv); OO = (nocc, nocc)
OOOO = (nocc, nocc, nocc, nocc); OVVO = (nocc, nv, nv, nocc); OVOV = (nocc, nv, nocc, nv)
OVOO = (nocc, nv, nocc, nocc); HVVDF = (naux, nv, nocc, nocc, nv)  # DF ladder intermediate
t1 = einsums.zeros((nocc, nv), name="t1")
t2 = E("i,j,a,b <- i,j,a,b ; i,j,a,b", G["oovv"], Dijab, SH2, 1.0, "t2init")  # placeholder, overwrite next
# MP2 guess t2 = oovv / Dijab (eager)
la.direct_division(1.0, G["oovv"], Dijab, 0.0, t2)

# ── deferred graph-owned scratch: declared on the GRAPH (not eager) so the
#    memory passes engage. g.declare_zero_tensor registers an AllocState::Deferred
#    handle with no Alloc node; MaterializationPass inserts Materialize+Initialize
#    (hoisted to the parent, allocated once before the loop) and MemoryPlanning can
#    then see the body's footprint. Eager create_zero_tensor stays opaque to both. ─
g = cg.Graph("ccsd")
def dz(name, shape): return g.declare_zero_tensor(name, list(shape), dtype="float64")
S = {}
for nm, sh in [("Tau", SH2), ("Taut", SH2), ("Fae", VV), ("Fmi", OO), ("Fme", NV2),
               ("Wmnij", OOOO), ("Wmbej", OVVO), ("Wmbje", OVOV), ("Zmbij", OVOO), ("jnfb", SH2),
               ("r1", NV2), ("r2", SH2), ("be", VV), ("jm", OO), ("imea", SH2), ("imeb", SH2),
               ("tmp", SH2), ("tmpP", SH2), ("rd1", NV2), ("rd2", SH2), ("Hvvdf", HVVDF)]:
    S[nm] = dz(nm, sh)
# scalars stay eager: cg.dot validates its result size at capture, and the
# predicate reads Ecorr between iterations — both need a live 1-element tensor.
e1 = zt("e1", (1,)); e2 = zt("e2", (1,)); Ecorr = zt("Ecorr", (1,))

def ein(spec, A, B, out, pf=1.0, acc=False):
    einsums.einsum(spec, out, A, B, c_pf=(1.0 if acc else 0.0), ab_pf=pf)

def symacc(spec, A, B, pf, sign=1.0):                # r2 += sign * sym(pf*(A⊗B))
    ein(spec, A, B, S["tmp"], pf, acc=False)
    la.axpby(sign, S["tmp"], 1.0, S["r2"])
    einsums.permute("j,i,b,a <- i,j,a,b", S["tmpP"], S["tmp"])
    la.axpby(sign, S["tmpP"], 1.0, S["r2"])

# ── CCSD iteration as a graph loop ───────────────────────────────────────────
e_prev = [1e9]
def cont(it):
    e = float(np.asarray(Ecorr)[0]); d = abs(e - e_prev[0]); e_prev[0] = e
    return (d > 1e-11) and (it < 99)

body = g.add_loop("ccsd_iter", 100, cont)
with cg.capture(body):
    Tau, Taut = S["Tau"], S["Taut"]
    ein("i,j,a,b <- i,a ; j,b", t1, t1, Tau, 1.0); la.axpby(1.0, t2, 1.0, Tau)
    ein("i,j,a,b <- i,a ; j,b", t1, t1, Taut, 0.5); la.axpby(1.0, t2, 1.0, Taut)

    Fae = S["Fae"]; la.axpby(1.0, Fvv, 0.0, Fae)
    ein("ae <- mf ; mafe", t1, G["ovvv"], Fae, 2.0, True)
    ein("ae <- mf ; maef", t1, G["ovvv"], Fae, -1.0, True)
    ein("ae <- mnaf ; mnef", Taut, G["oovv"], Fae, -2.0, True)
    ein("ae <- mnaf ; mnfe", Taut, G["oovv"], Fae, 1.0, True)
    Fmi = S["Fmi"]; la.axpby(1.0, Foo, 0.0, Fmi)
    ein("mi <- ne ; mnie", t1, G["ooov"], Fmi, 2.0, True)
    ein("mi <- ne ; mnei", t1, G["oovo"], Fmi, -1.0, True)
    ein("mi <- inef ; mnef", Taut, G["oovv"], Fmi, 2.0, True)
    ein("mi <- inef ; mnfe", Taut, G["oovv"], Fmi, -1.0, True)
    Fme = S["Fme"]
    ein("me <- nf ; mnef", t1, G["oovv"], Fme, 2.0, False)
    ein("me <- nf ; mnfe", t1, G["oovv"], Fme, -1.0, True)

    Wmnij = S["Wmnij"]; la.axpby(1.0, G["oooo"], 0.0, Wmnij)
    ein("mnij <- je ; mnie", t1, G["ooov"], Wmnij, 1.0, True)
    ein("mnij <- ie ; mnej", t1, G["oovo"], Wmnij, 1.0, True)
    ein("mnij <- ijef ; mnef", Tau, G["oovv"], Wmnij, 1.0, True)
    jnfb = S["jnfb"]
    ein("j,n,f,b <- j,f ; n,b", t1, t1, jnfb, 1.0, False); la.axpby(0.5, t2, 1.0, jnfb)
    Wmbej = S["Wmbej"]; la.axpby(1.0, G["ovvo"], 0.0, Wmbej)
    ein("mbej <- jf ; mbef", t1, G["ovvv"], Wmbej, 1.0, True)
    ein("mbej <- nb ; mnej", t1, G["oovo"], Wmbej, -1.0, True)
    ein("mbej <- jnfb ; mnef", jnfb, G["oovv"], Wmbej, -1.0, True)
    ein("mbej <- njfb ; mnef", t2, G["oovv"], Wmbej, 1.0, True)
    ein("mbej <- njfb ; mnfe", t2, G["oovv"], Wmbej, -0.5, True)
    Wmbje = S["Wmbje"]; la.axpby(-1.0, G["ovov"], 0.0, Wmbje)
    ein("mbje <- jf ; mbfe", t1, G["ovvv"], Wmbje, -1.0, True)
    ein("mbje <- nb ; mnje", t1, G["ooov"], Wmbje, 1.0, True)
    ein("mbje <- jnfb ; mnfe", jnfb, G["oovv"], Wmbje, 1.0, True)
    Zmbij = S["Zmbij"]
    ein("mbij <- mbef ; ijef", G["ovvv"], Tau, Zmbij, 1.0, False)

    # T1 residual
    r1 = S["r1"]
    ein("ia <- ie ; ae", t1, Fae, r1, 1.0, False)
    ein("ia <- ma ; mi", t1, Fmi, r1, -1.0, True)
    ein("ia <- imae ; me", t2, Fme, r1, 2.0, True)
    ein("ia <- imea ; me", t2, Fme, r1, -1.0, True)
    ein("ia <- nf ; nafi", t1, G["ovvo"], r1, 2.0, True)
    ein("ia <- nf ; naif", t1, G["ovov"], r1, -1.0, True)
    ein("ia <- mief ; maef", t2, G["ovvv"], r1, 2.0, True)
    ein("ia <- mife ; maef", t2, G["ovvv"], r1, -1.0, True)
    ein("ia <- mnae ; nmei", t2, G["oovo"], r1, -2.0, True)
    ein("ia <- mnae ; nmie", t2, G["ooov"], r1, 1.0, True)

    # T2 residual
    r2 = S["r2"]; la.axpby(1.0, G["oovv"], 0.0, r2)
    symacc("ijab <- ijae ; be", t2, Fae, 1.0, 1.0)
    ein("be <- mb ; me", t1, Fme, S["be"], 1.0, False)
    symacc("ijab <- ijae ; be", t2, S["be"], 0.5, -1.0)
    symacc("ijab <- imab ; mj", t2, Fmi, 1.0, -1.0)
    ein("jm <- je ; me", t1, Fme, S["jm"], 1.0, False)
    symacc("ijab <- imab ; jm", t2, S["jm"], 0.5, -1.0)
    ein("ijab <- mnab ; mnij", Tau, Wmnij, r2, 1.0, True)
    # DF particle-ladder (replaces ein("ijab <- ijef ; abef", Tau, G["vvvv"], ...)):
    #   r2_ijab += Σ_ef Tau_ijef <ab|ef>,  <ab|ef> = (ae|bf) = Σ_Q Bvv_ae Bvv_bf
    # two o²v³ steps via a DF intermediate H — never forms the v⁴ tensor.
    ein("Q,a,i,j,f <- Q,a,e ; i,j,e,f", Bvv, Tau, S["Hvvdf"], 1.0, False)
    ein("i,j,a,b <- Q,a,i,j,f ; Q,b,f", S["Hvvdf"], Bvv, r2, 1.0, True)
    symacc("ijab <- ma ; mbij", t1, Zmbij, 1.0, -1.0)
    # ring 1: sym((t_imae - t_imea)·Wmbej)
    ein("ijab <- imae ; mbej", t2, Wmbej, S["tmp"], 1.0, False)
    ein("ijab <- imea ; mbej", t2, Wmbej, S["tmp"], -1.0, True)
    la.axpby(1.0, S["tmp"], 1.0, r2); einsums.permute("j,i,b,a <- i,j,a,b", S["tmpP"], S["tmp"]); la.axpby(1.0, S["tmpP"], 1.0, r2)
    # ring 2: sym(t_imae·(Wmbej + Wmbje))
    ein("ijab <- imae ; mbej", t2, Wmbej, S["tmp"], 1.0, False)
    ein("ijab <- imae ; mbje", t2, Wmbje, S["tmp"], 1.0, True)
    la.axpby(1.0, S["tmp"], 1.0, r2); einsums.permute("j,i,b,a <- i,j,a,b", S["tmpP"], S["tmp"]); la.axpby(1.0, S["tmpP"], 1.0, r2)
    symacc("ijab <- mjae ; mbie", t2, Wmbje, 1.0, 1.0)
    ein("i,m,e,a <- i,e ; m,a", t1, t1, S["imea"], 1.0, False)
    symacc("ijab <- imea ; mbej", S["imea"], G["ovvo"], 1.0, -1.0)
    ein("i,m,e,b <- i,e ; m,b", t1, t1, S["imeb"], 1.0, False)
    symacc("ijab <- imeb ; maje", S["imeb"], G["ovov"], 1.0, -1.0)
    symacc("ijab <- ie ; abej", t1, G["vvvo"], 1.0, 1.0)
    symacc("ijab <- ma ; mbij", t1, G["ovoo"], 1.0, -1.0)

    # in-place amplitude update
    la.direct_division(1.0, r1, Dia, 0.0, S["rd1"]); la.axpby(1.0, S["rd1"], 1.0, t1)
    la.direct_division(1.0, r2, Dijab, 0.0, S["rd2"]); la.axpby(1.0, S["rd2"], 1.0, t2)

    # correlation energy -> Ecorr (drives the predicate)
    la.dot(e1, Tau, G["oovv"]); la.dot(e2, Tau, oovv_ba)
    la.axpby(2.0, e1, 0.0, Ecorr); la.axpby(-1.0, e2, 1.0, Ecorr)

print(f"captured loop body: {body.num_nodes()} nodes; parent {g.num_nodes()} nodes")

# LinearCombinationContractionFolding (opt-in): the spin-adaptation pairs above
# (2*g_X - g_Y where g_X,g_Y are the SAME integral read with transposed indices)
# fold into a single contraction r += A·(2*g - gᵀ). Recurses into the loop body.
import json as _json
def _contractions(gr):
    return sum(1 for n in _json.loads(gr.to_json()).get("nodes", []) if n.get("kind") in ("Einsum", "Gemm", "BatchedGemm"))
_before = _contractions(body)
_lccf = cg.PassManager(); _lccf.add(cg.LinearCombinationContractionFolding())
# _lccf.set_verbosity(3)
g.apply(_lccf)
print(f"LinearCombinationContractionFolding: body contractions {_before} -> {_contractions(body)}")

# The big scratch is graph-owned DEFERRED (g.declare_zero_tensor), so the memory
# passes now have work: MaterializationPass turns each deferred handle into a
# Materialize+Initialize pair HOISTED to the parent (allocated/zeroed once before
# the loop, not per iteration) — parent node count grows, body stays put. Passes
# also recurse into the body (each opts in via recurse_into_subgraphs), where
# Reorder reschedules; the scratch is reused in place so CSE/InplaceOptimization
# have nothing to fold. Correctness holds (vs psi4 conv CCSD).
mod = g.apply(cg.default_pass_manager())
print(f"optimized: modified={mod}, parent {g.num_nodes()} nodes (Materialize hoisted), body {body.num_nodes()} nodes")

g.execute()
e_new = float(np.asarray(Ecorr)[0])
print(f"hybrid DF-CCSD corr (DF v⁴, exact rest)    = {e_new:.10f}")
print(f"psi4 conv CCSD corr (exact v⁴)             = {ref:.10f}")
# The exact-v⁴ version of this graph matches conv CCSD to 1.26e-11; the ONLY change
# here is the v⁴ block -> its DF reconstruction (B⊗B), so the energy shifts by the
# DF v⁴ error. That shift must be small (good fit) yet nonzero (genuinely DF, not
# accidentally exact); the two-step factorization itself is exact to ~1e-13.
df_shift = abs(e_new - ref)
print(f"DF v⁴ shift vs conv CCSD                   = {df_shift:.2e}  (the v⁴-block DF approximation)")
assert 1e-6 < df_shift < 1e-3, df_shift
print("hybrid DF-CCSD (graph loop, DF v⁴ via B, exact bridge integrals elsewhere) OK")
