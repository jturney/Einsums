//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The psi4-world binary: one of the plugin's two shared objects, and the one
// that links psi4 and ONLY psi4. Its Einsums-world sibling is the stage module
// under cpp/, which links Einsums and only Einsums. One binary linking both is
// forbidden by the sealed-worlds rule; the two meet in the plugin's Python,
// crossing nothing but buffers.
//
// The division of labor: work that needs the host's machinery (basis sets,
// integrals, SCF quantities) happens here with psi4's API and is left on the
// wavefunction as plain Matrices; the correlated numerics happen in the stage
// module. Here that means one MO integral transform.

#include "psi4/liboptions/liboptions.h"
#include "psi4/libmints/matrix.h"
#include "psi4/libmints/mintshelper.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/psi4-dec.h"

namespace psi::hybrid_mp2 {

extern "C" PSI_API int read_options(std::string name, Options &options) {
    if (name == "HYBRID_MP2" || options.read_globals()) {
        /*- Which backend runs the contracted stages: the Python numerics or
            the compiled stage module (requires cpp/ to have been built). -*/
        options.add_str("STAGE_BACKEND", "PYTHON", "PYTHON CPP");
        /*- The amount of information printed to the output file -*/
        options.add_int("PRINT", 1);
    }
    return true;
}

// psi4's plugin ABI: an extern "C" entry point returning a C++ type. The
// warning about it is psi4's contract, not ours to fix.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"

extern "C" PSI_API SharedWavefunction hybrid_mp2(SharedWavefunction ref_wfn, Options &options) {
    if (!ref_wfn) throw PSIEXCEPTION("hybrid_mp2: SCF has not been run yet");

    // MO (ia|jb) over the active space, computed in the psi4 world. The
    // result is stashed on the wavefunction as an array variable, which is
    // the buffer-level handoff the Python adapter reads back.
    auto C_occ = ref_wfn->Ca_subset("AO", "ACTIVE_OCC");
    auto C_vir = ref_wfn->Ca_subset("AO", "ACTIVE_VIR");

    MintsHelper mints(ref_wfn->basisset(), options);
    auto iajb = mints.mo_eri(C_occ, C_vir, C_occ, C_vir);
    iajb->set_name("HYBRID_MP2 IAJB");
    ref_wfn->set_array_variable("HYBRID_MP2 IAJB", iajb);

    if (options.get_int("PRINT") > 0) {
        outfile->Printf("  hybrid_mp2: transformed (ia|jb) for %d occupied x %d virtual orbitals\n",
                        C_occ->colspi()[0], C_vir->colspi()[0]);
    }
    return ref_wfn;
}

#pragma clang diagnostic pop

}  // namespace psi::hybrid_mp2
