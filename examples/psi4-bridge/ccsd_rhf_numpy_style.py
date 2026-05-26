#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Closed-shell spin-adapted RHF-CCSD in einsums (eager, numpy-style).

Equations: D. Crawford's ccenergy form (psi4numpy helper_ccenergy.py), validated
in ccsd_rhf_oracle.py. This is the einsums realization, honoring two goals:
  * INTEGRALS FROM THE BRIDGE — all 10 physicist blocks <pq|rs>=(pr|qs) are built
    from MintsHelper.mo_bra_half_transform_einsums (3 half-transforms: oo/ov/vv)
    + a ket-finish IN EINSUMS + a chemist->physicist permute. No ao_eri.
  * EINSUMS, NOT NUMPY — every contraction/transform is einsums (einsum specs +
    operators + the '/' denominator + permute for the closed-shell symmetrizer);
    numpy ONLY ingests psi4 C/eps and reads the scalar energy.

This builds v4 EXPLICITLY (vvvv via the vv|vv half-transform) as a stepping
stone — the DF-CCSD production version swaps the v4-bearing terms (the τ·vvvv
term + Zmbij = <mb|ef>·τ) for direct B_vv contractions, never forming vvvv.

Validated: cc-pVDZ water, conventional SCF/CCSD. Matches psi4 to ~1e-11:
    closed-shell einsums CCSD corr = -0.2134804971

Run (Einsums build + psi4 stage on PYTHONPATH, conda-env Python)::

    PYTHONPATH=/Users/jturney/Code/Einsums/Einsums/build/lib:/Users/jturney/Code/psi4/cmake-build-debug/stage/lib \
        /Users/jturney/miniconda3/envs/einsums-dev/bin/python \
        examples/psi4-bridge/ccsd_rhf_numpy_style.py
"""
import numpy as np
import psi4
import einsums
import einsums._core
from einsums import linalg as la

psi4.core.set_output_file("/tmp/psi4_ccsd_conv.out", False)
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

def E(spec, A, B, shape, pf=1.0, name="x"):
    out = einsums.create_zero_tensor(name, list(shape), dtype="float64")
    einsums.einsum(spec, out, A, B, c_pf=0.0, ab_pf=pf)
    return out

def perm(spec, x, shape, name="p"):
    out = einsums.create_zero_tensor(name, list(shape), dtype="float64")
    einsums.permute(spec, out, x)
    return out

def sym(t):  # closed-shell symmetrizer r_ijab + r_jiba
    return t + perm("j,i,b,a <- i,j,a,b", t, (nocc, nocc, nv, nv), "sym")

def to_t(name, a): return einsums.asarray(np.ascontiguousarray(a), name=name)

# ── integrals from the bridge: physicist <pq|rs> = (pr|qs)_chem ──────────────
HT = {("o", "o"): mints.mo_bra_half_transform_einsums(Co, Co),
      ("o", "v"): mints.mo_bra_half_transform_einsums(Co, Cv),
      ("v", "v"): mints.mo_bra_half_transform_einsums(Cv, Cv)}
Ct = {"o": to_t("Co", Co_np), "v": to_t("Cv", Cv_np)}
nsp = {"o": nocc, "v": nv}

def phys(P, Q, R, S, name):
    ht = HT[(P, R)]; nP, nR, nQ, nS = nsp[P], nsp[R], nsp[Q], nsp[S]
    tmp = E("a,b,q,sig <- lam,q ; a,b,lam,sig", Ct[Q], ht, (nP, nR, nQ, nbf), name=name + "_t")
    chem = E("a,b,q,s <- sig,s ; a,b,q,sig", Ct[S], tmp, (nP, nR, nQ, nS), name=name + "_c")
    return perm("a,q,b,s <- a,b,q,s", chem, (nP, nQ, nR, nS), name)

specs = {"oovv": ("o", "o", "v", "v"), "oooo": ("o", "o", "o", "o"), "vvvv": ("v", "v", "v", "v"),
         "ovvo": ("o", "v", "v", "o"), "ovov": ("o", "v", "o", "v"), "ooov": ("o", "o", "o", "v"),
         "oovo": ("o", "o", "v", "o"), "ovvv": ("o", "v", "v", "v"), "vvvo": ("v", "v", "v", "o"),
         "ovoo": ("o", "v", "o", "o")}
g = {nm: phys(*pqrs, nm) for nm, pqrs in specs.items()}

# ── bare Fock (canonical RHF: Fov=0) + denominators (numpy ingest) ───────────
Fvv = to_t("Fvv", np.diag(ev)); Foo = to_t("Foo", np.diag(eo))
Dia = to_t("Dia", eo[:, None] - ev[None, :])
Dijab = to_t("Dijab", eo[:, None, None, None] + eo[None, :, None, None]
            - ev[None, None, :, None] - ev[None, None, None, :])
oovv_ba = perm("i,j,b,a <- i,j,a,b", g["oovv"], (nocc, nocc, nv, nv), "oovv_ba")

t1 = einsums.zeros((nocc, nv), name="t1")
t2 = g["oovv"] / Dijab

NV2 = (nocc, nv); SH2 = (nocc, nocc, nv, nv)
OOOO = (nocc, nocc, nocc, nocc); OVVO = (nocc, nv, nv, nocc); OVOV = (nocc, nv, nocc, nv)
def tau():  return t2 + E("i,j,a,b <- i,a ; j,b", t1, t1, SH2, 1.0, "ot")
def taut(): return t2 + E("i,j,a,b <- i,a ; j,b", t1, t1, SH2, 0.5, "ot5")

def energy():
    tt = tau()
    return 2.0 * float(la.dot(tt, g["oovv"])) - float(la.dot(tt, oovv_ba))

e_old = energy()
print(f"MP2 (closed-shell einsums) = {e_old:.10f}")
for it in range(100):
    Tau, Taut = tau(), taut()
    Fae = Fvv * 1.0 \
        + E("ae <- mf ; mafe", t1, g["ovvv"], (nv, nv), 2.0, "Fae1") \
        + E("ae <- mf ; maef", t1, g["ovvv"], (nv, nv), -1.0, "Fae2") \
        - E("ae <- mnaf ; mnef", Taut, g["oovv"], (nv, nv), 2.0, "Fae3") \
        - E("ae <- mnaf ; mnfe", Taut, g["oovv"], (nv, nv), -1.0, "Fae4")
    Fmi = Foo * 1.0 \
        + E("mi <- ne ; mnie", t1, g["ooov"], (nocc, nocc), 2.0, "Fmi1") \
        + E("mi <- ne ; mnei", t1, g["oovo"], (nocc, nocc), -1.0, "Fmi2") \
        + E("mi <- inef ; mnef", Taut, g["oovv"], (nocc, nocc), 2.0, "Fmi3") \
        + E("mi <- inef ; mnfe", Taut, g["oovv"], (nocc, nocc), -1.0, "Fmi4")
    Fme = E("me <- nf ; mnef", t1, g["oovv"], NV2, 2.0, "Fme1") \
        + E("me <- nf ; mnfe", t1, g["oovv"], NV2, -1.0, "Fme2")

    Wmnij = g["oooo"] * 1.0 \
        + E("mnij <- je ; mnie", t1, g["ooov"], OOOO, 1.0, "Wmn1") \
        + E("mnij <- ie ; mnej", t1, g["oovo"], OOOO, 1.0, "Wmn2") \
        + E("mnij <- ijef ; mnef", Tau, g["oovv"], OOOO, 1.0, "Wmn3")
    jnfb = t2 * 0.5 + E("j,n,f,b <- j,f ; n,b", t1, t1, SH2, 1.0, "jnfb")
    Wmbej = g["ovvo"] * 1.0 \
        + E("mbej <- jf ; mbef", t1, g["ovvv"], OVVO, 1.0, "Wmbej1") \
        - E("mbej <- nb ; mnej", t1, g["oovo"], OVVO, 1.0, "Wmbej2") \
        - E("mbej <- jnfb ; mnef", jnfb, g["oovv"], OVVO, 1.0, "Wmbej3") \
        + E("mbej <- njfb ; mnef", t2, g["oovv"], OVVO, 1.0, "Wmbej4") \
        - E("mbej <- njfb ; mnfe", t2, g["oovv"], OVVO, 0.5, "Wmbej5")
    Wmbje = g["ovov"] * (-1.0) \
        - E("mbje <- jf ; mbfe", t1, g["ovvv"], OVOV, 1.0, "Wmbje1") \
        + E("mbje <- nb ; mnje", t1, g["ooov"], OVOV, 1.0, "Wmbje2") \
        + E("mbje <- jnfb ; mnfe", jnfb, g["oovv"], OVOV, 1.0, "Wmbje3")
    Zmbij = E("mbij <- mbef ; ijef", g["ovvv"], Tau, (nocc, nv, nocc, nocc), 1.0, "Zmbij")

    r1 = E("ia <- ie ; ae", t1, Fae, NV2, 1.0, "r1a") \
        - E("ia <- ma ; mi", t1, Fmi, NV2, 1.0, "r1b") \
        + E("ia <- imae ; me", t2, Fme, NV2, 2.0, "r1c") \
        + E("ia <- imea ; me", t2, Fme, NV2, -1.0, "r1d") \
        + E("ia <- nf ; nafi", t1, g["ovvo"], NV2, 2.0, "r1e") \
        + E("ia <- nf ; naif", t1, g["ovov"], NV2, -1.0, "r1f") \
        + E("ia <- mief ; maef", t2, g["ovvv"], NV2, 2.0, "r1g") \
        + E("ia <- mife ; maef", t2, g["ovvv"], NV2, -1.0, "r1h") \
        - E("ia <- mnae ; nmei", t2, g["oovo"], NV2, 2.0, "r1i") \
        - E("ia <- mnae ; nmie", t2, g["ooov"], NV2, -1.0, "r1j")

    r2 = g["oovv"] * 1.0
    r2 = r2 + sym(E("ijab <- ijae ; be", t2, Fae, SH2, 1.0, "t2a"))
    be = E("be <- mb ; me", t1, Fme, (nv, nv), 1.0, "be")
    r2 = r2 - sym(E("ijab <- ijae ; be", t2, be, SH2, 0.5, "t2b"))
    r2 = r2 - sym(E("ijab <- imab ; mj", t2, Fmi, SH2, 1.0, "t2c"))
    jm = E("jm <- je ; me", t1, Fme, (nocc, nocc), 1.0, "jm")
    r2 = r2 - sym(E("ijab <- imab ; jm", t2, jm, SH2, 0.5, "t2d"))
    r2 = r2 + E("ijab <- mnab ; mnij", Tau, Wmnij, SH2, 1.0, "t2e")
    r2 = r2 + E("ijab <- ijef ; abef", Tau, g["vvvv"], SH2, 1.0, "t2f")    # explicit v4
    r2 = r2 - sym(E("ijab <- ma ; mbij", t1, Zmbij, SH2, 1.0, "t2g"))      # Zmbij uses v3 (ovvv)
    r2 = r2 + sym(E("ijab <- imae ; mbej", t2, Wmbej, SH2, 1.0, "rng1")
                  + E("ijab <- imea ; mbej", t2, Wmbej, SH2, -1.0, "rng2"))
    r2 = r2 + sym(E("ijab <- imae ; mbej", t2, Wmbej, SH2, 1.0, "rng3")
                  + E("ijab <- imae ; mbje", t2, Wmbje, SH2, 1.0, "rng4"))
    r2 = r2 + sym(E("ijab <- mjae ; mbie", t2, Wmbje, SH2, 1.0, "rng5"))
    imea = E("i,m,e,a <- i,e ; m,a", t1, t1, (nocc, nocc, nv, nv), 1.0, "imea")
    r2 = r2 - sym(E("ijab <- imea ; mbej", imea, g["ovvo"], SH2, 1.0, "t2h"))
    imeb = E("i,m,e,b <- i,e ; m,b", t1, t1, (nocc, nocc, nv, nv), 1.0, "imeb")
    r2 = r2 - sym(E("ijab <- imeb ; maje", imeb, g["ovov"], SH2, 1.0, "t2i"))
    r2 = r2 + sym(E("ijab <- ie ; abej", t1, g["vvvo"], SH2, 1.0, "t2j"))
    r2 = r2 - sym(E("ijab <- ma ; mbij", t1, g["ovoo"], SH2, 1.0, "t2k"))

    t1 = t1 + r1 / Dia
    t2 = t2 + r2 / Dijab
    e_new = energy()
    if abs(e_new - e_old) < 1e-11:
        print(f"converged in {it+1} iters"); break
    e_old = e_new

print(f"closed-shell einsums CCSD corr = {e_new:.10f}")
print(f"psi4 conv CCSD corr            = {ref:.10f}")
print(f"difference                     = {abs(e_new - ref):.2e}")
assert abs(e_new - ref) < 1e-7
print("closed-shell spin-adapted RHF-CCSD (eager einsums, bridge integrals, explicit v4) MATCHES psi4")
