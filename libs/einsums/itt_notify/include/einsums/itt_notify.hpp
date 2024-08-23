// ----------------------------------------------------------------------------------------------
//  Copyright (c) The Einsums Developers. All rights reserved.
//  Licensed under the MIT License. See LICENSE.txt in the project root for license information.
// ----------------------------------------------------------------------------------------------

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <einsums/config.hpp>

// NOLINTBEGIN(bugprone-reserved-identifier)
struct ___itt_caller;
struct ___itt_string_handle;
struct ___itt_domain;
struct ___itt_id;
using __itt_heap_function = void *;
struct ___itt_counter;
// NOLINTEND(bugprone-reserved-identifier)

///////////////////////////////////////////////////////////////////////////////
// decide whether to use the ITT notify API if it's available

// #if EINSUMS_HAVE_ITTNOTIFY != 0
EINSUMS_EXPORT extern bool use_ittnotify_api;

///////////////////////////////////////////////////////////////////////////////
EINSUMS_EXPORT void itt_sync_create(void *addr, const char *objtype, const char *objname) noexcept;
EINSUMS_EXPORT void itt_sync_rename(void *addr, const char *name) noexcept;
EINSUMS_EXPORT void itt_sync_prepare(void *addr) noexcept;
EINSUMS_EXPORT void itt_sync_acquired(void *addr) noexcept;
EINSUMS_EXPORT void itt_sync_cancel(void *addr) noexcept;
EINSUMS_EXPORT void itt_sync_releasing(void *addr) noexcept;
EINSUMS_EXPORT void itt_sync_released(void *addr) noexcept;
EINSUMS_EXPORT void itt_sync_destroy(void *addr) noexcept;

EINSUMS_EXPORT ___itt_caller *itt_stack_create() noexcept;
EINSUMS_EXPORT void           itt_stack_enter(___itt_caller *ctx) noexcept;
EINSUMS_EXPORT void           itt_stack_leave(___itt_caller *ctx) noexcept;
EINSUMS_EXPORT void           itt_stack_destroy(___itt_caller *ctx) noexcept;

EINSUMS_EXPORT void itt_frame_begin(___itt_domain const *frame, ___itt_id *id) noexcept;
EINSUMS_EXPORT void itt_frame_end(___itt_domain const *frame, ___itt_id *id) noexcept;

EINSUMS_EXPORT int  itt_mark_create(char const *) noexcept;
EINSUMS_EXPORT void itt_mark_off(int mark) noexcept;
EINSUMS_EXPORT void itt_mark(int mark, char const *) noexcept;

EINSUMS_EXPORT void itt_thread_set_name(char const *) noexcept;
EINSUMS_EXPORT void itt_thread_ignore() noexcept;

EINSUMS_EXPORT void itt_task_begin(___itt_domain const *, ___itt_string_handle *) noexcept;
EINSUMS_EXPORT void itt_task_begin(___itt_domain const *, ___itt_id *, ___itt_string_handle *) noexcept;
EINSUMS_EXPORT void itt_task_end(___itt_domain const *) noexcept;

EINSUMS_EXPORT ___itt_domain        *itt_domain_create(char const *) noexcept;
EINSUMS_EXPORT ___itt_string_handle *itt_string_handle_create(char const *) noexcept;

EINSUMS_EXPORT ___itt_id *itt_make_id(void *, std::size_t);
EINSUMS_EXPORT void       itt_id_create(___itt_domain const *, ___itt_id *id) noexcept;
EINSUMS_EXPORT void       itt_id_destroy(___itt_id *id) noexcept;

EINSUMS_EXPORT __itt_heap_function itt_heap_function_create(const char *, const char *) noexcept;
EINSUMS_EXPORT void                itt_heap_allocate_begin(__itt_heap_function, std::size_t, int) noexcept;
EINSUMS_EXPORT void                itt_heap_allocate_end(__itt_heap_function, void **, std::size_t, int) noexcept;
EINSUMS_EXPORT void                itt_heap_free_begin(__itt_heap_function, void *) noexcept;
EINSUMS_EXPORT void                itt_heap_free_end(__itt_heap_function, void *) noexcept;
EINSUMS_EXPORT void                itt_heap_reallocate_begin(__itt_heap_function, void *, std::size_t, int) noexcept;
EINSUMS_EXPORT void                itt_heap_reallocate_end(__itt_heap_function, void *, void **, std::size_t, int) noexcept;
EINSUMS_EXPORT void                itt_heap_internal_access_begin() noexcept;
EINSUMS_EXPORT void                itt_heap_internal_access_end() noexcept;

EINSUMS_EXPORT ___itt_counter *itt_counter_create(char const *, char const *) noexcept;
EINSUMS_EXPORT ___itt_counter *itt_counter_create_typed(char const *, char const *, int) noexcept;
EINSUMS_EXPORT void            itt_counter_destroy(___itt_counter *) noexcept;
EINSUMS_EXPORT void            itt_counter_set_value(___itt_counter *, void *) noexcept;

EINSUMS_EXPORT int itt_event_create(char const *name, int namelen) noexcept;
EINSUMS_EXPORT int itt_event_start(int evnt) noexcept;
EINSUMS_EXPORT int itt_event_end(int evnt) noexcept;

EINSUMS_EXPORT void itt_metadata_add(___itt_domain *domain, ___itt_id *id, ___itt_string_handle *key, std::uint64_t const &data) noexcept;
EINSUMS_EXPORT void itt_metadata_add(___itt_domain *domain, ___itt_id *id, ___itt_string_handle *key, double const &data) noexcept;
EINSUMS_EXPORT void itt_metadata_add(___itt_domain *domain, ___itt_id *id, ___itt_string_handle *key, char const *data) noexcept;
EINSUMS_EXPORT void itt_metadata_add(___itt_domain *domain, ___itt_id *id, ___itt_string_handle *key, void const *data) noexcept;

///////////////////////////////////////////////////////////////////////////////

namespace einsums {
namespace itt {

struct stack_context {
    EINSUMS_EXPORT stack_context();
    EINSUMS_EXPORT ~stack_context();

    stack_context(stack_context const &rhs) = delete;
    stack_context(stack_context &&rhs) noexcept : itt_context_(rhs.itt_context_) { rhs.itt_context_ = nullptr; }

    stack_context &operator=(stack_context const &rhs) = delete;
    stack_context &operator=(stack_context &&rhs) noexcept {
        if (this != &rhs) {
            itt_context_     = rhs.itt_context_;
            rhs.itt_context_ = nullptr;
        }
        return *this;
    }

    struct ___itt_caller *itt_context_ = nullptr;
};

struct caller_context {
    EINSUMS_EXPORT caller_context(stack_context &ctx);
    EINSUMS_EXPORT ~caller_context();

    stack_context &ctx_;
};

//////////////////////////////////////////////////////////////////////////
struct domain {
    EINSUMS_NON_COPYABLE(domain);

    domain() = default;
    EINSUMS_EXPORT domain(char const *) noexcept;

    ___itt_domain *domain_ = nullptr;
};

struct thread_domain : domain {
    EINSUMS_NON_COPYABLE(thread_domain);

    EINSUMS_EXPORT thread_domain() noexcept;
};

struct id {
    EINSUMS_EXPORT id(domain const &domain, void *addr, unsigned long extra = 0) noexcept;
    EINSUMS_EXPORT ~id();

    id(id const &rhs) = delete;
    id(id &&rhs) noexcept : id_(rhs.id_) { rhs.id_ = nullptr; }

    id &operator=(id const &rhs) = delete;
    id &operator=(id &&rhs) noexcept {
        if (this != &rhs) {
            id_     = rhs.id_;
            rhs.id_ = nullptr;
        }
        return *this;
    }

    ___itt_id *id_ = nullptr;
};

///////////////////////////////////////////////////////////////////////////
struct frame_context {
    EINSUMS_EXPORT frame_context(domain const &domain, id *ident = nullptr) noexcept;
    EINSUMS_EXPORT ~frame_context();

    domain const &domain_;
    id           *ident_ = nullptr;
};

struct undo_frame_context {
    EINSUMS_EXPORT undo_frame_context(frame_context &frame) noexcept;
    EINSUMS_EXPORT ~undo_frame_context();

    frame_context &frame_;
};

///////////////////////////////////////////////////////////////////////////
struct mark_context {
    EINSUMS_EXPORT mark_context(char const *name) noexcept;
    EINSUMS_EXPORT ~mark_context();

    int         itt_mark_;
    char const *name_ = nullptr;
};

struct undo_mark_context {
    EINSUMS_EXPORT undo_mark_context(mark_context &mark) noexcept;
    EINSUMS_EXPORT ~undo_mark_context();

    mark_context &mark_;
};

///////////////////////////////////////////////////////////////////////////
struct string_handle {
    string_handle() noexcept = default;

    EINSUMS_EXPORT string_handle(char const *s) noexcept;

    string_handle(___itt_string_handle *h) noexcept : handle_(h) {}

    string_handle(string_handle const &) = default;
    string_handle(string_handle &&rhs) : handle_(rhs.handle_) { rhs.handle_ = nullptr; }

    string_handle &operator=(string_handle const &) = default;
    string_handle &operator=(string_handle &&rhs) noexcept {
        if (this != &rhs) {
            handle_     = rhs.handle_;
            rhs.handle_ = nullptr;
        }
        return *this;
    }

    string_handle &operator=(___itt_string_handle *h) noexcept {
        handle_ = h;
        return *this;
    }

    explicit constexpr operator bool() const noexcept { return handle_ != nullptr; }

    ___itt_string_handle *handle_ = nullptr;
};

///////////////////////////////////////////////////////////////////////////
struct task {
    EINSUMS_EXPORT task(domain const &, string_handle const &, std::uint64_t metadata) noexcept;
    EINSUMS_EXPORT task(domain const &, string_handle const &) noexcept;
    EINSUMS_EXPORT ~task();

    EINSUMS_EXPORT void add_metadata(string_handle const &name, std::uint64_t val) noexcept;
    EINSUMS_EXPORT void add_metadata(string_handle const &name, double val) noexcept;
    EINSUMS_EXPORT void add_metadata(string_handle const &name, char const *val) noexcept;
    EINSUMS_EXPORT void add_metadata(string_handle const &name, void const *val) noexcept;

    template <typename T>
    void add_metadata(string_handle const &name, T const &val) noexcept {
        add_metadata(name, static_cast<void const *>(&val));
    }

    domain const &domain_;
    ___itt_id    *id_ = nullptr;
    string_handle sh_;
};

///////////////////////////////////////////////////////////////////////////
struct heap_function {
    EINSUMS_EXPORT heap_function(char const *name, char const *domain) noexcept;

    __itt_heap_function heap_function_ = nullptr;
};

struct heap_internal_access {
    EINSUMS_EXPORT heap_internal_access() noexcept;
    EINSUMS_EXPORT ~heap_internal_access();
};

struct heap_allocate {
    template <typename T>
    heap_allocate(heap_function &heap_function, T **&addr, std::size_t size, int init) noexcept
        : heap_function_(heap_function), addr_(reinterpret_cast<void **&>(addr)), size_(size), init_(init) {
        if (use_ittnotify_api) {
            itt_heap_allocate_begin(heap_function_.heap_function_, size_, init_);
        }
    }

    ~heap_allocate() {
        if (use_ittnotify_api) {
            itt_heap_allocate_end(heap_function_.heap_function_, addr_, size_, init_);
        }
    }

  private:
    heap_function &heap_function_;
    void        **&addr_;
    std::size_t    size_;
    int            init_;
};

struct heap_free {
    EINSUMS_EXPORT heap_free(heap_function &heap_function, void *addr) noexcept;
    EINSUMS_EXPORT ~heap_free();

  private:
    heap_function &heap_function_;
    void          *addr_;
};

///////////////////////////////////////////////////////////////////////////
struct counter {
    EINSUMS_EXPORT counter(char const *name, char const *domain) noexcept;
    EINSUMS_EXPORT counter(char const *name, char const *domain, int type) noexcept;
    EINSUMS_EXPORT ~counter();

    template <typename T>
    void set_value(T const &value) noexcept {
        if (use_ittnotify_api && id_) {
            itt_counter_set_value(id_, const_cast<void *>(static_cast<const void *>(&value)));
        }
    }

    counter(counter const &rhs) = delete;
    counter(counter &&rhs) noexcept : id_(rhs.id_) { rhs.id_ = nullptr; }

    counter &operator=(counter const &rhs) = delete;
    counter &operator=(counter &&rhs) noexcept {
        if (this != &rhs) {
            id_     = rhs.id_;
            rhs.id_ = nullptr;
        }
        return *this;
    }

  private:
    ___itt_counter *id_ = nullptr;
};

///////////////////////////////////////////////////////////////////////////
struct event {
    event(char const *name) noexcept : event_(itt_event_create(name, (int)strnlen(name, 256))) {}

    void start() const noexcept { itt_event_start(event_); }

    void end() const noexcept { itt_event_end(event_); }

  private:
    int event_ = 0;
};

struct mark_event {
    mark_event(event const &e) noexcept : e_(e) { e_.start(); }
    ~mark_event() { e_.end(); }

  private:
    event e_;
};

inline void event_tick(event const &e) noexcept {
    e.start();
}
} // namespace itt
} // namespace einsums

// #else
#if 0
inline constexpr void itt_sync_create(void *, const char *, const char *) noexcept {
}
inline constexpr void itt_sync_rename(void *, const char *) noexcept {
}
inline constexpr void itt_sync_prepare(void *) noexcept {
}
inline constexpr void itt_sync_acquired(void *) noexcept {
}
inline constexpr void itt_sync_cancel(void *) noexcept {
}
inline constexpr void itt_sync_releasing(void *) noexcept {
}
inline constexpr void itt_sync_released(void *) noexcept {
}
inline constexpr void itt_sync_destroy(void *) noexcept {
}

inline constexpr ___itt_caller *itt_stack_create() noexcept {
    return nullptr;
}
inline constexpr void itt_stack_enter(___itt_caller *) noexcept {
}
inline constexpr void itt_stack_leave(___itt_caller *) noexcept {
}
inline constexpr void itt_stack_destroy(___itt_caller *) noexcept {
}

inline constexpr void itt_frame_begin(___itt_domain const *, ___itt_id *) noexcept {
}
inline constexpr void itt_frame_end(___itt_domain const *, ___itt_id *) noexcept {
}

inline constexpr int itt_mark_create(char const *) noexcept {
    return 0;
}
inline constexpr void itt_mark_off(int) noexcept {
}
inline constexpr void itt_mark(int, char const *) noexcept {
}

inline constexpr void itt_thread_set_name(char const *) noexcept {
}
inline constexpr void itt_thread_ignore() noexcept {
}

inline constexpr void itt_task_begin(___itt_domain const *, ___itt_string_handle *) noexcept {
}
inline constexpr void itt_task_begin(___itt_domain const *, ___itt_id *, ___itt_string_handle *) noexcept {
}
inline constexpr void itt_task_end(___itt_domain const *) noexcept {
}

inline constexpr ___itt_domain *itt_domain_create(char const *) noexcept {
    return nullptr;
}
inline constexpr ___itt_string_handle *itt_string_handle_create(char const *) noexcept {
    return nullptr;
}

inline constexpr ___itt_id *itt_make_id(void *, unsigned long) {
    return nullptr;
}
inline constexpr void itt_id_create(___itt_domain const *, ___itt_id *) noexcept {
}
inline constexpr void itt_id_destroy(___itt_id *) noexcept {
}

inline constexpr __itt_heap_function itt_heap_function_create(const char *, const char *) noexcept {
    return nullptr;
}
inline constexpr void itt_heap_allocate_begin(__itt_heap_function, std::size_t, int) noexcept {
}
inline constexpr void itt_heap_allocate_end(__itt_heap_function, void **, std::size_t, int) noexcept {
}
inline constexpr void itt_heap_free_begin(__itt_heap_function, void *) noexcept {
}
inline constexpr void itt_heap_free_end(__itt_heap_function, void *) noexcept {
}
inline constexpr void itt_heap_reallocate_begin(__itt_heap_function, void *, std::size_t, int) noexcept {
}
inline constexpr void itt_heap_reallocate_end(__itt_heap_function, void *, void **, std::size_t, int) noexcept {
}
inline constexpr void itt_heap_internal_access_begin() noexcept {
}
inline constexpr void itt_heap_internal_access_end() noexcept {
}

inline constexpr ___itt_counter *itt_counter_create(char const *, char const *) noexcept {
    return nullptr;
}
inline constexpr ___itt_counter *itt_counter_create_typed(char const *, char const *, int) noexcept {
    return nullptr;
}
inline constexpr void itt_counter_destroy(___itt_counter *) noexcept {
}
inline constexpr void itt_counter_set_value(___itt_counter *, void *) noexcept {
}

inline constexpr int itt_event_create(char const *, int) noexcept {
    return 0;
}
inline constexpr int itt_event_start(int) noexcept {
    return 0;
}
inline constexpr int itt_event_end(int) noexcept {
    return 0;
}

inline constexpr void itt_metadata_add(___itt_domain *, ___itt_id *, ___itt_string_handle *, std::uint64_t const &) noexcept {
}
inline constexpr void itt_metadata_add(___itt_domain *, ___itt_id *, ___itt_string_handle *, double const &) noexcept {
}
inline constexpr void itt_metadata_add(___itt_domain *, ___itt_id *, ___itt_string_handle *, char const *) noexcept {
}
inline constexpr void itt_metadata_add(___itt_domain *, ___itt_id *, ___itt_string_handle *, void const *) noexcept {
}

//////////////////////////////////////////////////////////////////////////////

namespace einsums::itt {

struct stack_context {
    stack_context()  = default;
    ~stack_context() = default;
};

struct caller_context {
    constexpr caller_context(stack_context &) noexcept {}
    ~caller_context() = default;
};

//////////////////////////////////////////////////////////////////////////
struct domain {
    EINSUMS_NON_COPYABLE(domain);

    constexpr domain(char const *) noexcept {}
    domain() = default;
};

struct thread_domain : domain {
    EINSUMS_NON_COPYABLE(thread_domain);

    thread_domain() = default;
};

struct id {
    constexpr id(domain const &, void *, unsigned long = 0) noexcept {}
    ~id() = default;
};

///////////////////////////////////////////////////////////////////////////
struct frame_context {
    constexpr frame_context(domain const &, id * = nullptr) noexcept {}
    ~frame_context() = default;
};

struct undo_frame_context {
    constexpr undo_frame_context(frame_context const &) noexcept {}
    ~undo_frame_context() = default;
};

///////////////////////////////////////////////////////////////////////////
struct mark_context {
    constexpr mark_context(char const *) noexcept {}
    ~mark_context() = default;
};

struct undo_mark_context {
    constexpr undo_mark_context(mark_context const &) noexcept {}
    ~undo_mark_context() = default;
};

///////////////////////////////////////////////////////////////////////////
struct string_handle {
    constexpr string_handle(char const * = nullptr) noexcept {}
};

//////////////////////////////////////////////////////////////////////////
struct task {
    constexpr task(domain const &, string_handle const &, std::uint64_t) noexcept {}
    constexpr task(domain const &, string_handle const &) noexcept {}

    ~task() = default;
};

///////////////////////////////////////////////////////////////////////////
struct heap_function {
    constexpr heap_function(char const *, char const *) noexcept {}
    ~heap_function() = default;
};

struct heap_allocate {
    template <typename T>
    constexpr heap_allocate(heap_function & /*heap_function*/, T **, std::size_t, int) noexcept {}
    ~heap_allocate() = default;
};

struct heap_free {
    constexpr heap_free(heap_function & /*heap_function*/, void *) noexcept {}
    ~heap_free() = default;
};

struct heap_internal_access {
    heap_internal_access()  = default;
    ~heap_internal_access() = default;
};

struct counter {
    constexpr counter(char const * /*name*/, char const * /*domain*/) noexcept {}
    ~counter() = default;
};

struct event {
    constexpr event(char const *) noexcept {}
};

struct mark_event {
    constexpr mark_event(event const &) noexcept {}
    ~mark_event() = default;
};

inline constexpr void event_tick(event const &) noexcept {
}
} // namespace einsums::itt

#endif // EINSUMS_HAVE_ITTNOTIFY
