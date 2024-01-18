//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>

// NOLINGBEGIN(bugprone-reserved-identifier)
struct ___itt_caller;
struct ___itt_string_handle;
struct ___itt_domain;
struct ___itt_id;
using __itt_heap_function = void *;
struct ___itt_counter;
// NOLINTEND(bugprone-reserved-identifier)

///////////////////////////////////////////////////////////////////////////////
#define EINSUMS_ITT_SYNC_CREATE(obj, type, name) itt_sync_create(obj, type, name)
#define EINSUMS_ITT_SYNC_RENAME(obj, name)       itt_sync_rename(obj, name)
#define EINSUMS_ITT_SYNC_PREPARE(obj)            itt_sync_prepare(obj)
#define EINSUMS_ITT_SYNC_CANCEL(obj)             itt_sync_cancel(obj)
#define EINSUMS_ITT_SYNC_ACQUIRED(obj)           itt_sync_acquired(obj)
#define EINSUMS_ITT_SYNC_RELEASING(obj)          itt_sync_releasing(obj)
#define EINSUMS_ITT_SYNC_RELEASED(obj)           itt_sync_released(obj)
#define EINSUMS_ITT_SYNC_DESTROY(obj)            itt_sync_destroy(obj)

#define EINSUMS_ITT_STACK_CREATE(ctx)       ctx = itt_stack_create()
#define EINSUMS_ITT_STACK_CALLEE_ENTER(ctx) itt_stack_enter(ctx)
#define EINSUMS_ITT_STACK_CALLEE_LEAVE(ctx) itt_stack_leave(ctx)
#define EINSUMS_ITT_STACK_DESTROY(ctx)      itt_stack_destroy(ctx)

#define EINSUMS_ITT_FRAME_BEGIN(frame, id) itt_frame_begin(frame, id)
#define EINSUMS_ITT_FRAME_END(frame, id)   itt_frame_end(frame, id)

#define EINSUMS_ITT_MARK_CREATE(mark, name) mark = itt_mark_create(name)
#define EINSUMS_ITT_MARK_OFF(mark)          itt_mark_off(mark)
#define EINSUMS_ITT_MARK(mark, parameter)   itt_mark(mark, parameter)

#define EINSUMS_ITT_THREAD_SET_NAME(name) itt_thread_set_name(name)
#define EINSUMS_ITT_THREAD_IGNORE()       itt_thread_ignore()

#define EINSUMS_ITT_TASK_BEGIN(domain, name)        itt_task_begin(domain, name)
#define EINSUMS_ITT_TASK_BEGIN_ID(domain, id, name) itt_task_begin(domain, id, name)
#define EINSUMS_ITT_TASK_END(domain)                itt_task_end(domain)

#define EINSUMS_ITT_DOMAIN_CREATE(name)        itt_domain_create(name)
#define EINSUMS_ITT_STRING_HANDLE_CREATE(name) itt_string_handle_create(name)

#define EINSUMS_ITT_MAKE_ID(addr, extra)  itt_make_id(addr, extra)
#define EINSUMS_ITT_ID_CREATE(domain, id) itt_id_create(domain, id)
#define EINSUMS_ITT_ID_DESTROY(id)        itt_id_destroy(id)

#define EINSUMS_ITT_HEAP_FUNCTION_CREATE(name, domain)            itt_heap_function_create(name, domain)            /**/
#define EINSUMS_ITT_HEAP_ALLOCATE_BEGIN(f, size, initialized)     itt_heap_allocate_begin(f, size, initialized)     /**/
#define EINSUMS_ITT_HEAP_ALLOCATE_END(f, addr, size, initialized) itt_heap_allocate_end(f, addr, size, initialized) /**/
#define EINSUMS_ITT_HEAP_FREE_BEGIN(f, addr)                      itt_heap_free_begin(f, addr)
#define EINSUMS_ITT_HEAP_FREE_END(f, addr)                        itt_heap_free_end(f, addr)
#define EINSUMS_ITT_HEAP_REALLOCATE_BEGIN(f, addr, new_size, initialized)                                              \
    itt_heap_reallocate_begin(f, addr, new_size, initialized) /**/
#define EINSUMS_ITT_HEAP_REALLOCATE_END(f, addr, new_addr, new_size, initialized)                                      \
    itt_heap_reallocate_end(f, addr, new_addr, new_size, initialized) /**/
#define EINSUMS_ITT_HEAP_INTERNAL_ACCESS_BEGIN() itt_heap_internal_access_begin()
#define EINSUMS_ITT_HEAP_INTERNAL_ACCESS_END()   itt_heap_internal_access_end()

#define EINSUMS_ITT_COUNTER_CREATE(name, domain)             itt_counter_create(name, domain)             /**/
#define EINSUMS_ITT_COUNTER_CREATE_TYPED(name, domain, type) itt_counter_create_typed(name, domain, type) /**/
#define EINSUMS_ITT_COUNTER_SET_VALUE(id, value_ptr)         itt_counter_set_value(id, value_ptr)         /**/
#define EINSUMS_ITT_COUNTER_DESTROY(id)                      itt_counter_destroy(id)

#define EINSUMS_ITT_METADATA_ADD(domain, id, key, data) itt_metadata_add(domain, id, key, data) /**/

#if EINSUMS_HAVE_ITTNOTIFY != 0
EINSUMS_EXPORT extern bool use_ittnotify_api;

///////////////////////////////////////////////////////////////////////////////
EINSUMS_EXPORT void itt_sync_create(void *addr, char const *objtype, char const *objname) noexcept;
EINSUMS_EXPORT void itt_sync_rename(void *addr, char const *name) noexcept;
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

EINSUMS_EXPORT __itt_heap_function itt_heap_function_create(char const *, char const *) noexcept;
EINSUMS_EXPORT void                itt_heap_allocate_begin(__itt_heap_function, std::size_t, int) noexcept;
EINSUMS_EXPORT void                itt_heap_allocate_end(__itt_heap_function, void **, std::size_t, int) noexcept;
EINSUMS_EXPORT void                itt_heap_free_begin(__itt_heap_function, void *) noexcept;
EINSUMS_EXPORT void                itt_heap_free_end(__itt_heap_function, void *) noexcept;
EINSUMS_EXPORT void                itt_heap_reallocate_begin(__itt_heap_function, void *, std::size_t, int) noexcept;
EINSUMS_EXPORT void itt_heap_reallocate_end(__itt_heap_function, void *, void **, std::size_t, int) noexcept;
EINSUMS_EXPORT void itt_heap_internal_access_begin() noexcept;
EINSUMS_EXPORT void itt_heap_internal_access_end() noexcept;

EINSUMS_EXPORT ___itt_counter *itt_counter_create(char const *, char const *) noexcept;
EINSUMS_EXPORT ___itt_counter *itt_counter_create_typed(char const *, char const *, int) noexcept;
EINSUMS_EXPORT void            itt_counter_destroy(___itt_counter *) noexcept;
EINSUMS_EXPORT void            itt_counter_set_value(___itt_counter *, void *) noexcept;

EINSUMS_EXPORT int itt_event_create(char const *name, int namelen) noexcept;
EINSUMS_EXPORT int itt_event_start(int evnt) noexcept;
EINSUMS_EXPORT int itt_event_end(int evnt) noexcept;

EINSUMS_EXPORT void itt_metadata_add(___itt_domain *domain, ___itt_id *id, ___itt_string_handle *key,
                                     std::uint64_t const &data) noexcept;
EINSUMS_EXPORT void itt_metadata_add(___itt_domain *domain, ___itt_id *id, ___itt_string_handle *key,
                                     double const &data) noexcept;
EINSUMS_EXPORT void itt_metadata_add(___itt_domain *domain, ___itt_id *id, ___itt_string_handle *key,
                                     char const *data) noexcept;
EINSUMS_EXPORT void itt_metadata_add(___itt_domain *domain, ___itt_id *id, ___itt_string_handle *key,
                                     void const *data) noexcept;

///////////////////////////////////////////////////////////////////////////////
namespace einsums::detail {

struct ThreadDescription;

} // namespace einsums::detail

namespace einsums::util::itt {

struct StackContext {
    EINSUMS_EXPORT StackContext();
    EINSUMS_EXPORT ~StackContext();

    StackContext(StackContext const &rhs) = delete;
    StackContext(StackContext &&rhs) noexcept : itt_context_(rhs.itt_context_) { rhs.itt_context_ = nullptr; }

    auto operator=(StackContext const &rhs) -> StackContext & = delete;
    auto operator=(StackContext &&rhs) noexcept -> StackContext & {
        if (this != &rhs) {
            itt_context_     = rhs.itt_context_;
            rhs.itt_context_ = nullptr;
        }
        return *this;
    }

    struct ___itt_caller *itt_context_ = nullptr;
};

struct CallerContext {
    EINSUMS_EXPORT CallerContext(StackContext &ctx);
    EINSUMS_EXPORT ~CallerContext();

    StackContext &ctx_;
};

//////////////////////////////////////////////////////////////////////////
struct Domain {
    EINSUMS_NON_COPYABLE(Domain);

    Domain() = default;
    EINSUMS_EXPORT Domain(char const *) noexcept;

    ___itt_domain *domain_ = nullptr;
};

struct ThreadDomain : Domain {
    EINSUMS_NON_COPYABLE(ThreadDomain);

    EINSUMS_EXPORT ThreadDomain() noexcept;
};

struct Id {
    EINSUMS_EXPORT Id(Domain const &domain, void *addr, unsigned long extra = 0) noexcept;
    EINSUMS_EXPORT ~Id();

    Id(Id const &rhs) = delete;
    Id(Id &&rhs) noexcept : id_(rhs.id_) { rhs.id_ = nullptr; }

    auto operator=(Id const &rhs) -> Id & = delete;
    auto operator=(Id &&rhs) noexcept -> Id & {
        if (this != &rhs) {
            id_     = rhs.id_;
            rhs.id_ = nullptr;
        }
        return *this;
    }

    ___itt_id *id_ = nullptr;
};

///////////////////////////////////////////////////////////////////////////
struct FrameContext {
    EINSUMS_EXPORT FrameContext(Domain const &domain, Id *ident = nullptr) noexcept;
    EINSUMS_EXPORT ~FrameContext();

    Domain const &domain_;
    Id           *ident_ = nullptr;
};

struct UndoFrameContext {
    EINSUMS_EXPORT UndoFrameContext(FrameContext &frame) noexcept;
    EINSUMS_EXPORT ~UndoFrameContext();

    FrameContext &frame_;
};

///////////////////////////////////////////////////////////////////////////
struct MarkContext {
    EINSUMS_EXPORT MarkContext(char const *name) noexcept;
    EINSUMS_EXPORT ~MarkContext();

    int         itt_mark_;
    char const *name_ = nullptr;
};

struct UndoMarkContext {
    EINSUMS_EXPORT UndoMarkContext(MarkContext &mark) noexcept;
    EINSUMS_EXPORT ~UndoMarkContext();

    MarkContext &mark_;
};

///////////////////////////////////////////////////////////////////////////
struct StringHandle {
    StringHandle() noexcept = default;

    EINSUMS_EXPORT StringHandle(char const *s) noexcept;

    StringHandle(___itt_string_handle *h) noexcept : handle_(h) {}

    StringHandle(StringHandle const &) = default;
    StringHandle(StringHandle &&rhs) noexcept : handle_(rhs.handle_) { rhs.handle_ = nullptr; }

    auto operator=(StringHandle const &) -> StringHandle & = default;
    auto operator=(StringHandle &&rhs) noexcept -> StringHandle & {
        if (this != &rhs) {
            handle_     = rhs.handle_;
            rhs.handle_ = nullptr;
        }
        return *this;
    }

    auto operator=(___itt_string_handle *h) noexcept -> StringHandle & {
        handle_ = h;
        return *this;
    }

    explicit constexpr operator bool() const noexcept { return handle_ != nullptr; }

    ___itt_string_handle *handle_ = nullptr;
};

///////////////////////////////////////////////////////////////////////////
struct Task {
    EINSUMS_EXPORT Task(Domain const &, StringHandle const &, std::uint64_t metadata) noexcept;
    EINSUMS_EXPORT Task(Domain const &, StringHandle const &) noexcept;
    EINSUMS_EXPORT ~Task();

    EINSUMS_EXPORT void add_metadata(StringHandle const &name, std::uint64_t val) noexcept;
    EINSUMS_EXPORT void add_metadata(StringHandle const &name, double val) noexcept;
    EINSUMS_EXPORT void add_metadata(StringHandle const &name, char const *val) noexcept;
    EINSUMS_EXPORT void add_metadata(StringHandle const &name, void const *val) noexcept;

    template <typename T>
    void add_metadata(StringHandle const &name, T const &val) noexcept {
        add_metadata(name, static_cast<void const *>(&val));
    }

    Domain const &domain_;
    ___itt_id    *id_ = nullptr;
    StringHandle  sh_;
};

///////////////////////////////////////////////////////////////////////////
struct HeapFunction {
    EINSUMS_EXPORT HeapFunction(char const *name, char const *domain) noexcept;

    __itt_heap_function heap_function_ = nullptr;
};

struct HeapInternalAccess {
    EINSUMS_EXPORT HeapInternalAccess() noexcept;
    EINSUMS_EXPORT ~HeapInternalAccess();
};

struct HeapAllocate {
    template <typename T>
    HeapAllocate(HeapFunction &heap_function, T **&addr, std::size_t size, int init) noexcept
        : _heap_function(heap_function), _addr(reinterpret_cast<void **&>(addr)), _size(size), _init(init) {
        if (use_ittnotify_api) {
            EINSUMS_ITT_HEAP_ALLOCATE_BEGIN(_heap_function.heap_function_, _size, _init);
        }
    }

    ~HeapAllocate() {
        if (use_ittnotify_api) {
            EINSUMS_ITT_HEAP_ALLOCATE_END(_heap_function.heap_function_, _addr, _size, _init);
        }
    }

  private:
    HeapFunction &_heap_function;
    void       **&_addr;
    std::size_t   _size;
    int           _init;
};

struct HeapFree {
    EINSUMS_EXPORT HeapFree(HeapFunction &heap_function, void *addr) noexcept;
    EINSUMS_EXPORT ~HeapFree();

  private:
    HeapFunction &_heap_function;
    void         *_addr;
};

///////////////////////////////////////////////////////////////////////////
struct Counter {
    EINSUMS_EXPORT Counter(char const *name, char const *domain) noexcept;
    EINSUMS_EXPORT Counter(char const *name, char const *domain, int type) noexcept;
    EINSUMS_EXPORT ~Counter();

    template <typename T>
    void set_value(T const &value) noexcept {
        if (use_ittnotify_api && _id) {
            EINSUMS_ITT_COUNTER_SET_VALUE(_id, const_cast<void *>(static_cast<void const *>(&value)));
        }
    }

    Counter(Counter const &rhs) = delete;
    Counter(Counter &&rhs) noexcept : _id(rhs._id) { rhs._id = nullptr; }

    auto operator=(Counter const &rhs) -> Counter & = delete;
    auto operator=(Counter &&rhs) noexcept -> Counter & {
        if (this != &rhs) {
            _id     = rhs._id;
            rhs._id = nullptr;
        }
        return *this;
    }

  private:
    ___itt_counter *_id = nullptr;
};

///////////////////////////////////////////////////////////////////////////
struct Event {
    Event(char const *name) noexcept : _event(itt_event_create(name, (int)strnlen(name, 256))) {}

    void start() const noexcept { itt_event_start(_event); }

    void end() const noexcept { itt_event_end(_event); }

  private:
    int _event = 0;
};

struct MarkEvent {
    MarkEvent(Event const &e) noexcept : _e(e) { _e.start(); }
    ~MarkEvent() { _e.end(); }

  private:
    Event _e;
};

inline void event_tick(Event const &e) noexcept {
    e.start();
}
} // namespace einsums::util::itt

#else

inline constexpr void itt_sync_create(void *, char const *, char const *) noexcept {
}
inline constexpr void itt_sync_rename(void *, char const *) noexcept {
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

inline constexpr __itt_heap_function itt_heap_function_create(char const *, char const *) noexcept {
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

inline constexpr void itt_metadata_add(___itt_domain *, ___itt_id *, ___itt_string_handle *,
                                       std::uint64_t const &) noexcept {
}
inline constexpr void itt_metadata_add(___itt_domain *, ___itt_id *, ___itt_string_handle *, double const &) noexcept {
}
inline constexpr void itt_metadata_add(___itt_domain *, ___itt_id *, ___itt_string_handle *, char const *) noexcept {
}
inline constexpr void itt_metadata_add(___itt_domain *, ___itt_id *, ___itt_string_handle *, void const *) noexcept {
}

//////////////////////////////////////////////////////////////////////////////
namespace einsums::detail {

struct ThreadDescription;

} // namespace einsums::detail

namespace einsums::util::itt {

struct StackContext {
    StackContext()  = default;
    ~StackContext() = default;
};

struct CallerContext {
    constexpr CallerContext(StackContext &) noexcept {}
    ~CallerContext() = default;
};

//////////////////////////////////////////////////////////////////////////
struct Domain {
    EINSUMS_NON_COPYABLE(Domain);

    constexpr Domain(char const *) noexcept {}
    Domain() = default;
};

struct ThreadDomain : Domain {
    EINSUMS_NON_COPYABLE(ThreadDomain);

    ThreadDomain() = default;
};

struct Id {
    constexpr Id(Domain const &, void *, unsigned long = 0) noexcept {}
    ~Id() = default;
};

///////////////////////////////////////////////////////////////////////////
struct FrameContext {
    constexpr FrameContext(Domain const &, Id * = nullptr) noexcept {}
    ~FrameContext() = default;
};

struct UndoFrameContext {
    constexpr UndoFrameContext(FrameContext const &) noexcept {}
    ~UndoFrameContext() = default;
};

///////////////////////////////////////////////////////////////////////////
struct MarkContext {
    constexpr MarkContext(char const *) noexcept {}
    ~MarkContext() = default;
};

struct UndoMarkContext {
    constexpr UndoMarkContext(MarkContext const &) noexcept {}
    ~UndoMarkContext() = default;
};

///////////////////////////////////////////////////////////////////////////
struct StringHandle {
    constexpr StringHandle(char const * = nullptr) noexcept {}
};

//////////////////////////////////////////////////////////////////////////
struct Task {
    constexpr Task(Domain const &, StringHandle const &, std::uint64_t) noexcept {}
    constexpr Task(Domain const &, StringHandle const &) noexcept {}

    ~Task() = default;
};

///////////////////////////////////////////////////////////////////////////
struct HeapFunction {
    constexpr HeapFunction(char const *, char const *) noexcept {}
    ~HeapFunction() = default;
};

struct HeapAllocate {
    template <typename T>
    constexpr HeapAllocate(HeapFunction & /*heap_function*/, T **, std::size_t, int) noexcept {}
    ~HeapAllocate() = default;
};

struct HeapFree {
    constexpr HeapFree(HeapFunction & /*heap_function*/, void *) noexcept {}
    ~HeapFree() = default;
};

struct HeapInternalAccess {
    HeapInternalAccess()  = default;
    ~HeapInternalAccess() = default;
};

struct Counter {
    constexpr Counter(char const * /*name*/, char const * /*domain*/) noexcept {}
    ~Counter() = default;
};

struct Event {
    constexpr Event(char const *) noexcept {}
};

struct MarkEvent {
    constexpr MarkEvent(Event const &) noexcept {}
    ~MarkEvent() = default;
};

inline constexpr void event_tick(Event const &) noexcept {
}
} // namespace einsums::util::itt

#endif // EINSUMS_HAVE_ITTNOTIFY
