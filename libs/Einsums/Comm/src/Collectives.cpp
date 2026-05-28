//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Comm/Collectives.hpp>
#include <Einsums/Comm/Communicator.hpp>
#include <Einsums/Comm/Runtime.hpp>
#include <Einsums/Preprocessor/Unused.hpp>
#include <Einsums/Profile.hpp>

#include <cstring>

#if defined(EINSUMS_HAVE_MPI)
#    include <mpi.h>
#endif

namespace einsums::comm {

// ═══════════════════════════════════════════════════════════════════════════════
// Request implementation
// ═══════════════════════════════════════════════════════════════════════════════

struct Request::Impl {
#if defined(EINSUMS_HAVE_MPI)
    MPI_Request request{MPI_REQUEST_NULL};
#endif
    bool completed{false};
};

void Request::wait() {
    if (!_impl || _impl->completed)
        return;
#if defined(EINSUMS_HAVE_MPI)
    MPI_Wait(&impl_->request, MPI_STATUS_IGNORE);
#endif
    _impl->completed = true;
}

bool Request::test() {
    if (!_impl || _impl->completed)
        return true;
#if defined(EINSUMS_HAVE_MPI)
    int flag = 0;
    MPI_Test(&impl_->request, &flag, MPI_STATUS_IGNORE);
    if (flag)
        impl_->completed = true;
    return flag != 0;
#else
    _impl->completed = true;
    return true;
#endif
}

void *Request::native_handle() const {
#if defined(EINSUMS_HAVE_MPI)
    if (impl_)
        return &impl_->request;
#endif
    return nullptr;
}

Request::Request(std::shared_ptr<Impl> impl) : _impl(std::move(impl)) {
}

// ═══════════════════════════════════════════════════════════════════════════════
// MPI helpers
// ═══════════════════════════════════════════════════════════════════════════════

#if defined(EINSUMS_HAVE_MPI)

namespace {

template <Communicable T>
MPI_Datatype mpi_type_for() {
    if constexpr (std::same_as<T, float>)
        return MPI_FLOAT;
    else if constexpr (std::same_as<T, double>)
        return MPI_DOUBLE;
    else if constexpr (std::same_as<T, int>)
        return MPI_INT;
    else if constexpr (std::same_as<T, long>)
        return MPI_LONG;
    else if constexpr (std::same_as<T, long long>)
        return MPI_LONG_LONG;
    else if constexpr (std::same_as<T, unsigned>)
        return MPI_UNSIGNED;
    else if constexpr (std::same_as<T, std::complex<float>>)
        return MPI_C_FLOAT_COMPLEX;
    else if constexpr (std::same_as<T, std::complex<double>>)
        return MPI_C_DOUBLE_COMPLEX;
    else
        static_assert(sizeof(T) == 0, "Unsupported type for MPI communication");
}

MPI_Op mpi_op_for(ReduceOp op) {
    switch (op) {
    case ReduceOp::Sum:
        return MPI_SUM;
    case ReduceOp::Max:
        return MPI_MAX;
    case ReduceOp::Min:
        return MPI_MIN;
    case ReduceOp::Prod:
        return MPI_PROD;
    }
    return MPI_SUM;
}

MPI_Comm get_mpi_comm(Communicator const &comm) {
    auto *handle = comm.native_handle();
    return handle ? *static_cast<MPI_Comm *>(handle) : MPI_COMM_WORLD;
}

/// Check MPI return code and convert to expected.
expected<void, CommError> check_mpi(int err, char const *func) {
    if (err == MPI_SUCCESS)
        return {};
    char error_string[MPI_MAX_ERROR_STRING];
    int  len = 0;
    MPI_Error_string(err, error_string, &len);
    return unexpected(CommError::from_mpi(fmt::format("{}: {}", func, std::string(error_string, len)), err));
}

} // namespace

#endif // EINSUMS_HAVE_MPI

// ═══════════════════════════════════════════════════════════════════════════════
// Blocking collectives
// ═══════════════════════════════════════════════════════════════════════════════

template <Communicable T>
expected<void, CommError> allreduce(std::span<T const> send, std::span<T> recv, ReduceOp op, Communicator const &comm) {
    EINSUMS_UNUSED(comm)
    EINSUMS_UNUSED(op)

    LabeledSection0();
#if defined(EINSUMS_HAVE_MPI)
    return check_mpi(
        MPI_Allreduce(send.data(), recv.data(), static_cast<int>(send.size()), mpi_type_for<T>(), mpi_op_for(op), get_mpi_comm(comm)),
        "MPI_Allreduce");
#else
    if (send.data() != recv.data() && !send.empty()) {
        std::memcpy(recv.data(), send.data(), send.size_bytes());
    }
    return {};
#endif
}

template <Communicable T>
expected<void, CommError> allreduce_inplace(std::span<T> buf, ReduceOp op, Communicator const &comm) {
    LabeledSection0();
#if defined(EINSUMS_HAVE_MPI)
    return check_mpi(
        MPI_Allreduce(MPI_IN_PLACE, buf.data(), static_cast<int>(buf.size()), mpi_type_for<T>(), mpi_op_for(op), get_mpi_comm(comm)),
        "MPI_Allreduce(in-place)");
#else
    (void)buf;
    (void)op;
    (void)comm;
    return {};
#endif
}

template <Communicable T>
expected<void, CommError> broadcast(std::span<T> buf, int root, Communicator const &comm) {
    LabeledSection0();
#if defined(EINSUMS_HAVE_MPI)
    return check_mpi(MPI_Bcast(buf.data(), static_cast<int>(buf.size()), mpi_type_for<T>(), root, get_mpi_comm(comm)), "MPI_Bcast");
#else
    (void)buf;
    (void)root;
    (void)comm;
    return {};
#endif
}

template <Communicable T>
expected<void, CommError> allgather(std::span<T const> send, std::span<T> recv, Communicator const &comm) {
    EINSUMS_UNUSED(comm)

#if defined(EINSUMS_HAVE_MPI)
    return check_mpi(MPI_Allgather(send.data(), static_cast<int>(send.size()), mpi_type_for<T>(), recv.data(),
                                   static_cast<int>(send.size()), mpi_type_for<T>(), get_mpi_comm(comm)),
                     "MPI_Allgather");
#else
    if (send.data() != recv.data() && !send.empty()) {
        std::memcpy(recv.data(), send.data(), send.size_bytes());
    }
    return {};
#endif
}

template <Communicable T>
expected<void, CommError> scatter(std::span<T const> send, std::span<T> recv, int root, Communicator const &comm) {
#if defined(EINSUMS_HAVE_MPI)
    return check_mpi(MPI_Scatter(send.data(), static_cast<int>(recv.size()), mpi_type_for<T>(), recv.data(), static_cast<int>(recv.size()),
                                 mpi_type_for<T>(), root, get_mpi_comm(comm)),
                     "MPI_Scatter");
#else
    if (send.data() != recv.data() && !recv.empty()) {
        std::memcpy(recv.data(), send.data(), recv.size_bytes());
    }
    (void)root;
    (void)comm;
    return {};
#endif
}

template <Communicable T>
expected<void, CommError> send(std::span<T const> buf, int dest, int tag, Communicator const &comm) {
#if defined(EINSUMS_HAVE_MPI)
    return check_mpi(MPI_Send(buf.data(), static_cast<int>(buf.size()), mpi_type_for<T>(), dest, tag, get_mpi_comm(comm)), "MPI_Send");
#else
    (void)buf;
    (void)dest;
    (void)tag;
    (void)comm;
    return {};
#endif
}

template <Communicable T>
expected<void, CommError> recv(std::span<T> buf, int src, int tag, Communicator const &comm) {
#if defined(EINSUMS_HAVE_MPI)
    return check_mpi(MPI_Recv(buf.data(), static_cast<int>(buf.size()), mpi_type_for<T>(), src, tag, get_mpi_comm(comm), MPI_STATUS_IGNORE),
                     "MPI_Recv");
#else
    (void)buf;
    (void)src;
    (void)tag;
    (void)comm;
    return {};
#endif
}

expected<void, CommError> barrier(Communicator const &comm) {
    LabeledSection0();
#if defined(EINSUMS_HAVE_MPI)
    return check_mpi(MPI_Barrier(get_mpi_comm(comm)), "MPI_Barrier");
#else
    (void)comm;
    return {};
#endif
}

// ═══════════════════════════════════════════════════════════════════════════════
// Non-blocking collectives
// ═══════════════════════════════════════════════════════════════════════════════

template <Communicable T>
expected<Request, CommError> iallreduce_inplace(std::span<T> buf, ReduceOp op, Communicator const &comm) {
    auto impl = std::make_shared<Request::Impl>();
#if defined(EINSUMS_HAVE_MPI)
    int err = MPI_Iallreduce(MPI_IN_PLACE, buf.data(), static_cast<int>(buf.size()), mpi_type_for<T>(), mpi_op_for(op), get_mpi_comm(comm),
                             &impl->request);
    if (err != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING];
        int  len = 0;
        MPI_Error_string(err, error_string, &len);
        return unexpected(CommError::from_mpi(fmt::format("MPI_Iallreduce(in-place): {}", std::string(error_string, len)), err));
    }
#else
    (void)buf;
    (void)op;
    (void)comm;
    impl->completed = true;
#endif
    return Request(std::move(impl));
}

template <Communicable T>
expected<Request, CommError> iallreduce(std::span<T const> send, std::span<T> recv, ReduceOp op, Communicator const &comm) {
    EINSUMS_UNUSED(op)
    EINSUMS_UNUSED(comm)

    auto impl = std::make_shared<Request::Impl>();
#if defined(EINSUMS_HAVE_MPI)
    int err = MPI_Iallreduce(send.data(), recv.data(), static_cast<int>(send.size()), mpi_type_for<T>(), mpi_op_for(op), get_mpi_comm(comm),
                             &impl->request);
    if (err != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING];
        int  len = 0;
        MPI_Error_string(err, error_string, &len);
        return unexpected(CommError::from_mpi(fmt::format("MPI_Iallreduce: {}", std::string(error_string, len)), err));
    }
#else
    if (send.data() != recv.data() && !send.empty()) {
        std::memcpy(recv.data(), send.data(), send.size_bytes());
    }
    impl->completed = true;
#endif
    return Request(std::move(impl));
}

template <Communicable T>
expected<Request, CommError> ibroadcast(std::span<T> buf, int root, Communicator const &comm) {
    auto impl = std::make_shared<Request::Impl>();
#if defined(EINSUMS_HAVE_MPI)
    int err = MPI_Ibcast(buf.data(), static_cast<int>(buf.size()), mpi_type_for<T>(), root, get_mpi_comm(comm), &impl->request);
    if (err != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING];
        int  len = 0;
        MPI_Error_string(err, error_string, &len);
        return unexpected(CommError::from_mpi(fmt::format("MPI_Ibcast: {}", std::string(error_string, len)), err));
    }
#else
    (void)buf;
    (void)root;
    (void)comm;
    impl->completed = true;
#endif
    return Request(std::move(impl));
}

template <Communicable T>
expected<Request, CommError> iallgather(std::span<T const> send, std::span<T> recv, Communicator const &comm) {
    EINSUMS_UNUSED(comm)

    auto impl = std::make_shared<Request::Impl>();
#if defined(EINSUMS_HAVE_MPI)
    int err = MPI_Iallgather(send.data(), static_cast<int>(send.size()), mpi_type_for<T>(), recv.data(), static_cast<int>(send.size()),
                             mpi_type_for<T>(), get_mpi_comm(comm), &impl->request);
    if (err != MPI_SUCCESS) {
        char error_string[MPI_MAX_ERROR_STRING];
        int  len = 0;
        MPI_Error_string(err, error_string, &len);
        return unexpected(CommError::from_mpi(fmt::format("MPI_Iallgather: {}", std::string(error_string, len)), err));
    }
#else
    if (send.data() != recv.data() && !send.empty()) {
        std::memcpy(recv.data(), send.data(), send.size_bytes());
    }
    impl->completed = true;
#endif
    return Request(std::move(impl));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Explicit template instantiations
// ═══════════════════════════════════════════════════════════════════════════════

#define INSTANTIATE_COLLECTIVES(T)                                                                                                         \
    template expected<void, CommError>    allreduce<T>(std::span<T const>, std::span<T>, ReduceOp, Communicator const &);                  \
    template expected<void, CommError>    allreduce_inplace<T>(std::span<T>, ReduceOp, Communicator const &);                              \
    template expected<void, CommError>    broadcast<T>(std::span<T>, int, Communicator const &);                                           \
    template expected<void, CommError>    allgather<T>(std::span<T const>, std::span<T>, Communicator const &);                            \
    template expected<void, CommError>    scatter<T>(std::span<T const>, std::span<T>, int, Communicator const &);                         \
    template expected<void, CommError>    send<T>(std::span<T const>, int, int, Communicator const &);                                     \
    template expected<void, CommError>    recv<T>(std::span<T>, int, int, Communicator const &);                                           \
    template expected<Request, CommError> iallreduce<T>(std::span<T const>, std::span<T>, ReduceOp, Communicator const &);                 \
    template expected<Request, CommError> iallreduce_inplace<T>(std::span<T>, ReduceOp, Communicator const &);                             \
    template expected<Request, CommError> ibroadcast<T>(std::span<T>, int, Communicator const &);                                          \
    template expected<Request, CommError> iallgather<T>(std::span<T const>, std::span<T>, Communicator const &);

// These definitions match the ``extern template EINSUMS_EXPORT ...`` declarations
// in Collectives.hpp, which carry the visibility attribute (needed because GCC
// and Linux Clang don't propagate visibility from the primary template to its
// explicit instantiations under EINSUMS_WITH_HIDDEN_VISIBILITY=ON).
INSTANTIATE_COLLECTIVES(float)
INSTANTIATE_COLLECTIVES(double)
INSTANTIATE_COLLECTIVES(int)
INSTANTIATE_COLLECTIVES(long)
INSTANTIATE_COLLECTIVES(long long)
INSTANTIATE_COLLECTIVES(std::complex<float>)
INSTANTIATE_COLLECTIVES(std::complex<double>)

#undef INSTANTIATE_COLLECTIVES

} // namespace einsums::comm
