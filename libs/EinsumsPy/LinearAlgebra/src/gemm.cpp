#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Utilities/InCollection.hpp>

#include <EinsumsPy/LinearAlgebra/LinearAlgebra.hpp>
#include <pybind11/buffer_info.h>
#include <pybind11/detail/common.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace einsums {
namespace python {
namespace detail {

void gemm(std::string const &transA, std::string const &transB, py::object const &alpha, pybind11::buffer const &A,
          pybind11::buffer const &B, py::object const &beta, pybind11::buffer &C) {
    py::buffer_info A_info = A.request(false), B_info = B.request(false), C_info = C.request(true);

    if (A_info.ndim != 2 || B_info.ndim != 2 || C_info.ndim != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "A call to gemm can only take rank-2 tensors as input!");
    }

    blas::int_t A_m, A_k, lda = A_info.strides[0] / A_info.itemsize;
    blas::int_t B_k, B_n, ldb = B_info.strides[0] / B_info.itemsize;
    blas::int_t C_m = C_info.shape[0], C_n = C_info.shape[1], ldc = C_info.strides[0] / C_info.itemsize;

    char transA_ch = 'n';

    if (transA.length() >= 1) {
        if (is_in(transA[0], {'n', 'N', 't', 'T', 'c', 'C'})) {
            transA_ch = transA[0];
        }
    }

    char transB_ch = 'n';

    if (transB.length() >= 1) {
        if (is_in(transB[0], {'n', 'N', 't', 'T', 'c', 'C'})) {
            transB_ch = transB[0];
        }
    }

    if (not_in(transA_ch, {'n', 'N'})) {
        A_m = A_info.shape[1];
        A_k = A_info.shape[0];
    } else {
        A_m = A_info.shape[0];
        A_k = A_info.shape[1];
    }

    if (not_in(transB_ch, {'n', 'N'})) {
        B_n = B_info.shape[0];
        B_k = B_info.shape[1];
    } else {
        B_n = B_info.shape[1];
        B_k = B_info.shape[0];
    }

    if (A_m != C_m || A_k != B_k || B_n != C_n) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The rows and columns of the matrices passed into gemm are not compatible!");
    }

    if (A_info.format != B_info.format || A_info.format != C_info.format) {
        EINSUMS_THROW_EXCEPTION(py::value_error, "Can only perform gemm on matrices with the same stored type! Got A ({}), B ({}), C ({}).",
                                A_info.format, B_info.format, C_info.format);
    }

    if (A_info.format == py::format_descriptor<float>::format()) {
        blas::gemm<float>(transA_ch, transB_ch, A_m, B_n, A_k, alpha.cast<float>(), (float const *)A_info.ptr, lda,
                          (float const *)B_info.ptr, ldb, beta.cast<float>(), (float *)C_info.ptr, ldc);
    } else if (A_info.format == py::format_descriptor<double>::format()) {
        blas::gemm<double>(transA_ch, transB_ch, A_m, B_n, A_k, alpha.cast<double>(), (double const *)A_info.ptr, lda,
                           (double const *)B_info.ptr, ldb, beta.cast<double>(), (double *)C_info.ptr, ldc);
    } else if (A_info.format == py::format_descriptor<std::complex<float>>::format()) {
        blas::gemm<std::complex<float>>(transA_ch, transB_ch, A_m, B_n, A_k, alpha.cast<std::complex<float>>(),
                                        (std::complex<float> const *)A_info.ptr, lda, (std::complex<float> const *)B_info.ptr, ldb,
                                        beta.cast<std::complex<float>>(), (std::complex<float> *)C_info.ptr, ldc);
    } else if (A_info.format == py::format_descriptor<std::complex<double>>::format()) {
        blas::gemm<std::complex<double>>(transA_ch, transB_ch, A_m, B_n, A_k, alpha.cast<std::complex<double>>(),
                                         (std::complex<double> const *)A_info.ptr, lda, (std::complex<double> const *)B_info.ptr, ldb,
                                         beta.cast<std::complex<double>>(), (std::complex<double> *)C_info.ptr, ldc);
    } else {
        EINSUMS_THROW_EXCEPTION(py::value_error, "Can only perform gemm on floating point matrices! Got type {}.", A_info.format);
    }
}

} // namespace detail
} // namespace python
} // namespace einsums