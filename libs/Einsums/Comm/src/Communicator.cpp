//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Comm/Communicator.hpp>
#include <Einsums/Comm/Runtime.hpp>

#if defined(EINSUMS_HAVE_MPI)
#    include <mpi.h>
#endif

namespace einsums::comm {

// ── Impl ────────────────────────────────────────────────────────────────────

struct Communicator::Impl {
#if defined(EINSUMS_HAVE_MPI)
    MPI_Comm comm{MPI_COMM_NULL};
    bool     owns{false}; // true if we should MPI_Comm_free on destruction

    Impl() = default;
    explicit Impl(MPI_Comm c, bool own = false) : comm(c), owns(own) {}
    ~Impl() {
        if (owns && comm != MPI_COMM_NULL && comm != MPI_COMM_WORLD && is_initialized()) {
            MPI_Comm_free(&comm);
        }
    }

    Impl(Impl const &)            = delete;
    Impl &operator=(Impl const &) = delete;
#else
    // Mock: nothing to store
#endif
};

// ── Communicator ────────────────────────────────────────────────────────────

Communicator &Communicator::world() {
    static auto impl = []() {
#if defined(EINSUMS_HAVE_MPI)
        return std::make_shared<Impl>(MPI_COMM_WORLD, /*owns=*/false);
#else
        return std::make_shared<Impl>();
#endif
    }();
    static Communicator world_comm(impl);
    return world_comm;
}

int Communicator::rank() const {
#if defined(EINSUMS_HAVE_MPI)
    if (impl_) {
        int r = 0;
        MPI_Comm_rank(impl_->comm, &r);
        return r;
    }
#endif
    return 0;
}

int Communicator::size() const {
#if defined(EINSUMS_HAVE_MPI)
    if (impl_) {
        int s = 1;
        MPI_Comm_size(impl_->comm, &s);
        return s;
    }
#endif
    return 1;
}

Communicator Communicator::split(int color, int key) const {
#if defined(EINSUMS_HAVE_MPI)
    if (impl_) {
        MPI_Comm new_comm = MPI_COMM_NULL;
        MPI_Comm_split(impl_->comm, color, key, &new_comm);
        return Communicator(std::make_shared<Impl>(new_comm, /*owns=*/true));
    }
#else
    (void)color;
    (void)key;
#endif
    return Communicator(std::make_shared<Impl>());
}

Communicator Communicator::dup() const {
#if defined(EINSUMS_HAVE_MPI)
    if (impl_) {
        MPI_Comm new_comm = MPI_COMM_NULL;
        MPI_Comm_dup(impl_->comm, &new_comm);
        return Communicator(std::make_shared<Impl>(new_comm, /*owns=*/true));
    }
#endif
    return Communicator(std::make_shared<Impl>());
}

void *Communicator::native_handle() const {
#if defined(EINSUMS_HAVE_MPI)
    if (impl_)
        return &impl_->comm;
#endif
    return nullptr;
}

} // namespace einsums::comm
