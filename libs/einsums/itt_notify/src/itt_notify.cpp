// ----------------------------------------------------------------------------------------------
//  Copyright (c) The Einsums Developers. All rights reserved.
//  Licensed under the MIT License. See LICENSE.txt in the project root for license information.
// ----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>
#include <einsums/itt_notify.hpp>

#if defined(EINSUMS_HAVE_ITTNOTIFY)

#    define INTEL_ITTNOTIFY_API_PRIVATE

#    include <cstddef>
#    include <cstdint>
#    include <ittnotify.h>
#    include <legacy/ittnotify.h>
#    include <memory>

///////////////////////////////////////////////////////////////////////////////
// decide whether to use the ITT notify API if it's available
bool use_ittnotify_api = false;

///////////////////////////////////////////////////////////////////////////////
#    define EINSUMS_INTERNAL_ITT_SYNC_CREATE(obj, type, name)                                                                              \
        if (use_ittnotify_api && __itt_sync_create_ptr) {                                                                                  \
            __itt_sync_create_ptr(const_cast<void *>(static_cast<volatile void *>(obj)), type, name, __itt_attr_mutex);                    \
        }                                                                                                                                  \
        /**/
#    define EINSUMS_INTERNAL_ITT_SYNC(fname, obj)                                                                                          \
        if (use_ittnotify_api && __itt_##fname##_ptr) {                                                                                    \
            __itt_##fname##_ptr(const_cast<void *>(static_cast<volatile void *>(obj)));                                                    \
        }                                                                                                                                  \
        /**/
#    define EINSUMS_INTERNAL_ITT_SYNC_RENAME(obj, name)                                                                                    \
        if (use_ittnotify_api && __itt_sync_rename_ptr) {                                                                                  \
            __itt_sync_rename_ptr(const_cast<void *>(static_cast<volatile void *>(obj)), name);                                            \
        }                                                                                                                                  \
        /**/

///////////////////////////////////////////////////////////////////////////////
#    define EINSUMS_INTERNAL_ITT_STACK_CREATE()                                                                                            \
        (use_ittnotify_api && __itt_stack_caller_create_ptr) ? __itt_stack_caller_create_ptr() : (__itt_caller) nullptr /**/
#    define EINSUMS_INTERNAL_ITT_STACK_ENTER(ctx)                                                                                          \
        if (use_ittnotify_api && __itt_stack_callee_enter_ptr)                                                                             \
            __itt_stack_callee_enter_ptr(ctx);                                                                                             \
        /**/
#    define EINSUMS_INTERNAL_ITT_STACK_LEAVE(ctx)                                                                                          \
        if (use_ittnotify_api && __itt_stack_callee_leave_ptr)                                                                             \
            __itt_stack_callee_leave_ptr(ctx);                                                                                             \
        /**/
#    define EINSUMS_INTERNAL_ITT_STACK_DESTROY(ctx)                                                                                        \
        if (use_ittnotify_api && __itt_stack_caller_destroy_ptr && ctx != (__itt_caller) nullptr)                                          \
            __itt_stack_caller_destroy_ptr(ctx);                                                                                           \
        /**/

///////////////////////////////////////////////////////////////////////////////
#    define EINSUMS_INTERNAL_ITT_FRAME_BEGIN(domain, id)                                                                                   \
        if (use_ittnotify_api && __itt_frame_begin_v3_ptr)                                                                                 \
            __itt_frame_begin_v3_ptr(domain, id);                                                                                          \
        /**/
#    define EINSUMS_INTERNAL_ITT_FRAME_END(domain, id)                                                                                     \
        if (use_ittnotify_api && __itt_frame_end_v3_ptr)                                                                                   \
            __itt_frame_end_v3_ptr(domain, id);                                                                                            \
        /**/

///////////////////////////////////////////////////////////////////////////////
#    define EINSUMS_INTERNAL_ITT_MARK_CREATE(name) (use_ittnotify_api && __itt_mark_create_ptr) ? __itt_mark_create_ptr(name) : 0 /**/
#    define EINSUMS_INTERNAL_ITT_MARK_OFF(mark)                                                                                            \
        if (use_ittnotify_api && __itt_mark_off_ptr)                                                                                       \
            __itt_mark_off_ptr(mark);                                                                                                      \
        /**/
#    define EINSUMS_INTERNAL_ITT_MARK(mark, parameter)                                                                                     \
        if (use_ittnotify_api && __itt_mark_ptr)                                                                                           \
            __itt_mark_ptr(mark, parameter);                                                                                               \
        /**/

///////////////////////////////////////////////////////////////////////////////
#    define EINSUMS_INTERNAL_ITT_THREAD_SET_NAME(name)                                                                                     \
        if (use_ittnotify_api && __itt_thread_set_name_ptr)                                                                                \
            __itt_thread_set_name_ptr(name);                                                                                               \
        /**/
#    define EINSUMS_INTERNAL_ITT_THREAD_IGNORE()                                                                                           \
        if (use_ittnotify_api && __itt_thread_ignore_ptr)                                                                                  \
            __itt_thread_ignore_ptr();                                                                                                     \
        /**/

///////////////////////////////////////////////////////////////////////////////
#    define EINSUMS_INTERNAL_ITT_TASK_BEGIN(domain, name)                                                                                  \
        if (use_ittnotify_api && __itt_task_begin_ptr)                                                                                     \
            __itt_task_begin_ptr(domain, __itt_null, __itt_null, name);                                                                    \
        /**/
#    define EINSUMS_INTERNAL_ITT_TASK_BEGIN_ID(domain, id, name)                                                                           \
        if (use_ittnotify_api && __itt_task_begin_ptr)                                                                                     \
            __itt_task_begin_ptr(domain, id, __itt_null, name);                                                                            \
        /**/
#    define EINSUMS_INTERNAL_ITT_TASK_END(domain)                                                                                          \
        if (use_ittnotify_api && __itt_task_end_ptr)                                                                                       \
            __itt_task_end_ptr(domain);                                                                                                    \
        /**/

#    define EINSUMS_INTERNAL_ITT_DOMAIN_CREATE(name)                                                                                       \
        (use_ittnotify_api && __itt_domain_create_ptr) ? __itt_domain_create_ptr(name) : nullptr /**/

#    define EINSUMS_INTERNAL_ITT_STRING_HANDLE_CREATE(name)                                                                                \
        (use_ittnotify_api && __itt_string_handle_create_ptr) ? __itt_string_handle_create_ptr(name) : nullptr /**/

#    define EINSUMS_INTERNAL_ITT_MAKE_ID(addr, extra) use_ittnotify_api ? __itt_id_make(addr, extra) : __itt_null /**/

#    define EINSUMS_INTERNAL_ITT_ID_CREATE(domain, id)                                                                                     \
        if (use_ittnotify_api && __itt_id_create_ptr)                                                                                      \
            __itt_id_create_ptr(domain, id);                                                                                               \
        /**/
#    define EINSUMS_INTERNAL_ITT_ID_DESTROY(id) delete id

///////////////////////////////////////////////////////////////////////////////
#    define EINSUMS_INTERNAL_ITT_HEAP_FUNCTION_CREATE(name, domain)                                                                        \
        (use_ittnotify_api && __itt_heap_function_create_ptr) ? __itt_heap_function_create_ptr(name, domain) : nullptr /**/

#    define EINSUMS_INTERNAL_HEAP_ALLOCATE_BEGIN(f, size, init)                                                                            \
        if (use_ittnotify_api && __itt_heap_allocate_begin_ptr)                                                                            \
            __itt_heap_allocate_begin_ptr(f, size, init);                                                                                  \
        /**/
#    define EINSUMS_INTERNAL_HEAP_ALLOCATE_END(f, addr, size, init)                                                                        \
        if (use_ittnotify_api && __itt_heap_allocate_end_ptr)                                                                              \
            __itt_heap_allocate_end_ptr(f, addr, size, init);                                                                              \
        /**/

#    define EINSUMS_INTERNAL_HEAP_FREE_BEGIN(f, addr)                                                                                      \
        if (use_ittnotify_api && __itt_heap_free_begin_ptr)                                                                                \
            __itt_heap_free_begin_ptr(f, addr);                                                                                            \
        /**/
#    define EINSUMS_INTERNAL_HEAP_FREE_END(f, addr)                                                                                        \
        if (use_ittnotify_api && __itt_heap_free_end_ptr)                                                                                  \
            __itt_heap_free_end_ptr(f, addr);                                                                                              \
        /**/

#    define EINSUMS_INTERNAL_HEAP_REALLOCATE_BEGIN(f, addr, size, init)                                                                    \
        if (use_ittnotify_api && __itt_heap_reallocate_begin_ptr)                                                                          \
            __itt_heap_reallocate_begin_ptr(f, addr, size, init);                                                                          \
        /**/
#    define EINSUMS_INTERNAL_HEAP_REALLOCATE_END(f, addr, new_addr, size, init)                                                            \
        if (use_ittnotify_api && __itt_heap_reallocate_end_ptr)                                                                            \
            __itt_heap_reallocate_end_ptr(f, addr, new_addr, size, init);                                                                  \
        /**/

#    define EINSUMS_INTERNAL_INTERNAL_ACCESS_BEGIN()                                                                                       \
        if (use_ittnotify_api && __itt_heap_internal_access_begin_ptr)                                                                     \
            __itt_heap_internal_access_begin_ptr();                                                                                        \
        /**/
#    define EINSUMS_INTERNAL_INTERNAL_ACCESS_END()                                                                                         \
        if (use_ittnotify_api && __itt_heap_internal_access_end_ptr)                                                                       \
            __itt_heap_internal_access_end_ptr();                                                                                          \
        /**/

///////////////////////////////////////////////////////////////////////////////
#    if defined(__itt_counter_create_typed_ptr) && defined(__itt_counter_set_value_ptr)
#        define EINSUMS_INTERNAL_COUNTER_CREATE(name, domain)                                                                              \
            (use_ittnotify_api && __itt_counter_create_ptr) ? __itt_counter_create_ptr(name, domain) : (__itt_counter) nullptr /**/
#        define EINSUMS_INTERNAL_COUNTER_CREATE_TYPED(name, domain, type)                                                                  \
            (use_ittnotify_api && __itt_counter_create_typed_ptr) ? __itt_counter_create_typed_ptr(name, domain, type)                     \
                                                                  : (__itt_counter) nullptr /**/
#        define EINSUMS_INTERNAL_COUNTER_SET_VALUE(id, value_ptr)                                                                          \
            if (use_ittnotify_api && __itt_counter_set_value_ptr)                                                                          \
                __itt_counter_set_value_ptr(id, value_ptr);                                                                                \
            /**/
#        define EINSUMS_INTERNAL_COUNTER_DESTROY(id)                                                                                       \
            if (use_ittnotify_api && __itt_counter_destroy_ptr)                                                                            \
                __itt_counter_destroy_ptr(id);                                                                                             \
            /**/
#    else
// older itt-notify implementations don't support the typed counter API
#        define EINSUMS_INTERNAL_COUNTER_CREATE(name, domain)             (__itt_counter) nullptr /**/
#        define EINSUMS_INTERNAL_COUNTER_CREATE_TYPED(name, domain, type) (__itt_counter) nullptr /**/
#        define EINSUMS_INTERNAL_COUNTER_SET_VALUE(id, value_ptr)                                 /**/
#        define EINSUMS_INTERNAL_COUNTER_DESTROY(id)                                              /**/
#    endif

///////////////////////////////////////////////////////////////////////////////
#    define EINSUMS_INTERNAL_EVENT_CREATE(name, len)                                                                                       \
        (use_ittnotify_api && __itt_event_create_ptr) ? __itt_event_create_ptr(name, len) : 0;                                             \
        /**/
#    define EINSUMS_INTERNAL_EVENT_START(e) (use_ittnotify_api && __itt_event_start_ptr) ? __itt_event_start_ptr(e) : 0 /**/
#    define EINSUMS_INTERNAL_EVENT_END(e)   (use_ittnotify_api && __itt_event_end_ptr) ? __itt_event_end_ptr(e) : 0     /**/

///////////////////////////////////////////////////////////////////////////////
#    define EINSUMS_INTERNAL_METADATA_ADD(domain, id, key, type, count, data)                                                              \
        if (use_ittnotify_api && __itt_metadata_add_ptr)                                                                                   \
        __itt_metadata_add_ptr(domain, id, key, type, count, data) /**/
#    define EINSUMS_INTERNAL_METADATA_STR_ADD(domain, id, key, data)                                                                       \
        if (use_ittnotify_api && __itt_metadata_str_add_ptr)                                                                               \
        __itt_metadata_str_add_ptr(domain, id, key, data, 0) /**/

///////////////////////////////////////////////////////////////////////////////
#    if defined(EINSUMS_MSVC) || defined(__BORLANDC__) || (defined(__MWERKS__) && defined(_WIN32) && (__MWERKS__ >= 0x3000)) ||            \
        (defined(__ICL) && defined(_MSC_EXTENSIONS) && (EINSUMS_MSVC >= 1200))

#        pragma comment(lib, "libittnotify.lib")
#    endif

///////////////////////////////////////////////////////////////////////////////
#    define EINSUMS_INTERNAL_ITT_SYNC_PREPARE(obj)   EINSUMS_INTERNAL_ITT_SYNC(sync_prepare, obj)
#    define EINSUMS_INTERNAL_ITT_SYNC_CANCEL(obj)    EINSUMS_INTERNAL_ITT_SYNC(sync_cancel, obj)
#    define EINSUMS_INTERNAL_ITT_SYNC_ACQUIRED(obj)  EINSUMS_INTERNAL_ITT_SYNC(sync_acquired, obj)
#    define EINSUMS_INTERNAL_ITT_SYNC_RELEASING(obj) EINSUMS_INTERNAL_ITT_SYNC(sync_releasing, obj)
#    define EINSUMS_INTERNAL_ITT_SYNC_RELEASED(obj)  ((void)0) // EINSUMS_INTERNAL_ITT_SYNC(sync_released, obj)
#    define EINSUMS_INTERNAL_ITT_SYNC_DESTROY(obj)   EINSUMS_INTERNAL_ITT_SYNC(sync_destroy, obj)

namespace einsums::itt {

stack_context::stack_context() {
    _itt_context = itt_stack_create();
}

stack_context::~stack_context() {
    if (_itt_context) {
        itt_stack_destroy(_itt_context);
    }
}

caller_context::caller_context(stack_context &ctx) : _ctx(ctx) {
    if (_ctx._itt_context) {
        itt_stack_enter(_ctx._itt_context);
    }
}

caller_context::~caller_context() {
    if (_ctx._itt_context) {
        itt_stack_leave(_ctx._itt_context);
    }
}

domain::domain(char const *name) noexcept : _domain(itt_domain_create(name)) {
    if (_domain != nullptr) {
        _domain->flags = 1;
    }
}

static thread_local std::unique_ptr<___itt_domain> _thread_domain;

thread_domain::thread_domain() noexcept : domain() {
    if (_thread_domain == nullptr) {
        // TODO: Fix this
        _thread_domain.reset(itt_domain_create("master"));
    }

    _domain = _thread_domain.get();

    if (_domain != nullptr) {
        _domain->flags = 1;
    }
}

id::id(domain const &domain, void *addr, unsigned long extra) noexcept {
    if (use_ittnotify_api) {
        _id = itt_make_id(addr, extra);
        itt_id_create(domain._domain, _id);
    }
}

id::~id() {
    if (use_ittnotify_api) {
        itt_id_destroy(_id);
    }
}

frame_context::frame_context(domain const &domain, id *ident) noexcept : _domain(domain), _ident(ident) {
    itt_frame_begin(_domain._domain, _ident ? _ident->_id : nullptr);
}

frame_context::~frame_context() {
    itt_frame_end(_domain._domain, _ident ? _ident->_id : nullptr);
}

undo_frame_context::undo_frame_context(frame_context &frame) noexcept : _frame(frame) {
    itt_frame_end(_frame._domain._domain, nullptr);
}

undo_frame_context::~undo_frame_context() {
    itt_frame_end(_frame._domain._domain, nullptr);
}

mark_context::mark_context(char const *name) noexcept : _itt_mark(0), _name(name) {
    _itt_mark = itt_mark_create(name);
}

mark_context::~mark_context() {
    itt_mark_off(_itt_mark);
}

undo_mark_context::undo_mark_context(mark_context &mark) noexcept : _mark(mark) {
    itt_mark_off(_mark._itt_mark);
}
undo_mark_context::~undo_mark_context() {
    _mark._itt_mark = itt_mark_create(_mark._name);
}

string_handle::string_handle(char const *s) noexcept : _handle(s == nullptr ? nullptr : itt_string_handle_create(s)) {
}

task::task(domain const &domain, string_handle const &name) noexcept : _domain(domain), _sh(name) {
    if (use_ittnotify_api) {
        _id = itt_make_id(_domain._domain, reinterpret_cast<std::size_t>(_sh._handle));

        itt_task_begin(_domain._domain, _id, _sh._handle);
    }
}

task::task(domain const &domain, string_handle const &name, std::uint64_t metadata) noexcept : _domain(domain), _sh(name) {
    if (use_ittnotify_api) {
        _id = itt_make_id(_domain._domain, reinterpret_cast<std::size_t>(_sh._handle));

        itt_task_begin(_domain._domain, _id, _sh._handle);
        add_metadata(_sh, metadata);
    }
}

task::~task() {
    if (use_ittnotify_api) {
        itt_task_end(_domain._domain);
        delete _id;
    }
}

void task::add_metadata(string_handle const &name, std::uint64_t val) noexcept {
    itt_metadata_add(_domain._domain, _id, name._handle, val);
}

void task::add_metadata(string_handle const &name, double val) noexcept {
    itt_metadata_add(_domain._domain, _id, name._handle, val);
}

void task::add_metadata(string_handle const &name, char const *val) noexcept {
    itt_metadata_add(_domain._domain, _id, name._handle, val);
}

void task::add_metadata(string_handle const &name, void const *val) noexcept {
    itt_metadata_add(_domain._domain, _id, name._handle, val);
}

heap_function::heap_function(char const *name, char const *domain) noexcept : _heap_function(itt_heap_function_create(name, domain)) {
}

heap_internal_access::heap_internal_access() noexcept {
    itt_heap_internal_access_begin();
}

heap_internal_access::~heap_internal_access() {
    itt_heap_internal_access_end();
}

heap_free::heap_free(heap_function &heap_function, void *addr) noexcept : _heap_function(heap_function), _addr(addr) {
    itt_heap_free_begin(_heap_function._heap_function, _addr);
}

heap_free::~heap_free() {
    itt_heap_free_end(_heap_function._heap_function, _addr);
}

counter::counter(char const *name, char const *domain) noexcept : _id(itt_counter_create(name, domain)) {
}

counter::counter(char const *name, char const *domain, int type) noexcept : _id(itt_counter_create_typed(name, domain, type)) {
}

counter::~counter() {
    if (_id) {
        itt_counter_destroy(_id);
    }
}
} // namespace einsums::itt

///////////////////////////////////////////////////////////////////////////////
void itt_sync_create(void *addr, const char *objtype, const char *objname) noexcept {
    EINSUMS_INTERNAL_ITT_SYNC_CREATE(addr, objtype, objname);
}

void itt_sync_rename(void *addr, const char *objname) noexcept {
    EINSUMS_INTERNAL_ITT_SYNC_RENAME(addr, objname);
}

void itt_sync_prepare(void *addr) noexcept {
    EINSUMS_INTERNAL_ITT_SYNC_PREPARE(addr);
}

void itt_sync_acquired(void *addr) noexcept {
    EINSUMS_INTERNAL_ITT_SYNC_ACQUIRED(addr);
}

void itt_sync_cancel(void *addr) noexcept {
    EINSUMS_INTERNAL_ITT_SYNC_CANCEL(addr);
}

void itt_sync_releasing(void *addr) noexcept {
    EINSUMS_INTERNAL_ITT_SYNC_RELEASING(addr);
}

void itt_sync_released(void *addr) noexcept {
    EINSUMS_INTERNAL_ITT_SYNC_RELEASED(addr);
}

void itt_sync_destroy(void *addr) noexcept {
    EINSUMS_INTERNAL_ITT_SYNC_DESTROY(addr);
}

///////////////////////////////////////////////////////////////////////////////
__itt_caller itt_stack_create() noexcept {
    return EINSUMS_INTERNAL_ITT_STACK_CREATE();
}

void itt_stack_enter(__itt_caller ctx) noexcept {
    EINSUMS_INTERNAL_ITT_STACK_ENTER(ctx);
}

void itt_stack_leave(__itt_caller ctx) noexcept {
    EINSUMS_INTERNAL_ITT_STACK_LEAVE(ctx);
}

void itt_stack_destroy(__itt_caller ctx) noexcept {
    EINSUMS_INTERNAL_ITT_STACK_DESTROY(ctx);
}

///////////////////////////////////////////////////////////////////////////////
void itt_frame_begin(___itt_domain const *domain, ___itt_id *id) noexcept {
    EINSUMS_INTERNAL_ITT_FRAME_BEGIN(domain, id);
}

void itt_frame_end(___itt_domain const *domain, ___itt_id *id) noexcept {
    EINSUMS_INTERNAL_ITT_FRAME_END(domain, id);
}

///////////////////////////////////////////////////////////////////////////////
int itt_mark_create(char const *name) noexcept {
    return EINSUMS_INTERNAL_ITT_MARK_CREATE(name);
}

void itt_mark_off(int mark) noexcept {
    EINSUMS_INTERNAL_ITT_MARK_OFF(mark);
}

void itt_mark(int mark, char const *par) noexcept {
    EINSUMS_INTERNAL_ITT_MARK(mark, par);
}

///////////////////////////////////////////////////////////////////////////////
void itt_thread_set_name(char const *name) noexcept {
    EINSUMS_INTERNAL_ITT_THREAD_SET_NAME(name);
}

void itt_thread_ignore() noexcept {
    EINSUMS_INTERNAL_ITT_THREAD_IGNORE();
}

///////////////////////////////////////////////////////////////////////////////
void itt_task_begin(___itt_domain const *domain, ___itt_string_handle *name) noexcept {
    EINSUMS_INTERNAL_ITT_TASK_BEGIN(domain, name);
}

void itt_task_begin(___itt_domain const *domain, ___itt_id *id, ___itt_string_handle *name) noexcept {
    EINSUMS_INTERNAL_ITT_TASK_BEGIN_ID(domain, *id, name);
}

void itt_task_end(___itt_domain const *domain) noexcept {
    EINSUMS_INTERNAL_ITT_TASK_END(domain);
}

___itt_domain *itt__domaincreate(char const *name) noexcept {
    return EINSUMS_INTERNAL_ITT_DOMAIN_CREATE(name);
}

___itt_string_handle *itt_string__handlecreate(char const *name) noexcept {
    return EINSUMS_INTERNAL_ITT_STRING_HANDLE_CREATE(name);
}

___itt_id *itt_make_id(void *addr, std::size_t extra) {
    return new ___itt_id(EINSUMS_INTERNAL_ITT_MAKE_ID(addr, extra));
}

void itt__idcreate(___itt_domain const *domain, ___itt_id *id) noexcept {
    EINSUMS_INTERNAL_ITT_ID_CREATE(domain, *id);
}

void itt__iddestroy(___itt_id *id) noexcept {
    EINSUMS_INTERNAL_ITT_ID_DESTROY(id);
}

///////////////////////////////////////////////////////////////////////////////
__itt_heap_function itt_heap_function_create(const char *name, const char *domain) noexcept {
    return EINSUMS_INTERNAL_ITT_HEAP_FUNCTION_CREATE(name, domain);
}

void itt_heap_allocate_begin(__itt_heap_function f, std::size_t size, int init) noexcept {
    EINSUMS_INTERNAL_HEAP_ALLOCATE_BEGIN(f, size, init);
}

void itt_heap_allocate_end(__itt_heap_function f, void **addr, std::size_t size, int init) noexcept {
    EINSUMS_INTERNAL_HEAP_ALLOCATE_END(f, addr, size, init);
}

void itt_heap_free_begin(__itt_heap_function f, void *addr) noexcept {
    EINSUMS_INTERNAL_HEAP_FREE_BEGIN(f, addr);
}

void itt_heap_free_end(__itt_heap_function f, void *addr) noexcept {
    EINSUMS_INTERNAL_HEAP_FREE_END(f, addr);
}

void itt_heap_reallocate_begin(__itt_heap_function f, void *addr, std::size_t new_size, int init) noexcept {
    EINSUMS_INTERNAL_HEAP_REALLOCATE_BEGIN(f, addr, new_size, init);
}

void itt_heap_reallocate_end(__itt_heap_function f, void *addr, void **new_addr, std::size_t new_size, int init) noexcept {
    EINSUMS_INTERNAL_HEAP_REALLOCATE_END(f, addr, new_addr, new_size, init);
}

void itt_heap_internal_access_begin() noexcept {
    EINSUMS_INTERNAL_INTERNAL_ACCESS_BEGIN();
}

void itt_heap_internal_access_end() noexcept {
    EINSUMS_INTERNAL_INTERNAL_ACCESS_END();
}

///////////////////////////////////////////////////////////////////////////////
__itt_counter itt_counter_create(char const *name, char const *domain) noexcept {
    return EINSUMS_INTERNAL_COUNTER_CREATE(name, domain);
}

__itt_counter itt_counter_create_typed(char const *name, char const *domain, int type) noexcept {
    return EINSUMS_INTERNAL_COUNTER_CREATE_TYPED(name, domain, (__itt_metadata_type)type);
}

void itt_counter_destroy(__itt_counter id) noexcept {
    EINSUMS_INTERNAL_COUNTER_DESTROY(id);
}

void itt_counter_set_value(__itt_counter id, void *value_ptr) noexcept {
    EINSUMS_INTERNAL_COUNTER_SET_VALUE(id, value_ptr);
}

///////////////////////////////////////////////////////////////////////////////
__itt_event itt_event_create(char const *name, int namelen) noexcept {
    return EINSUMS_INTERNAL_EVENT_CREATE(name, namelen);
}

int itt_event_start(__itt_event evnt) noexcept {
    return EINSUMS_INTERNAL_EVENT_START(evnt);
}

int itt_event_end(__itt_event evnt) noexcept {
    return EINSUMS_INTERNAL_EVENT_END(evnt);
}

///////////////////////////////////////////////////////////////////////////////
void itt_metadata_add(___itt_domain *domain, ___itt_id *id, ___itt_string_handle *key, std::uint64_t const &data) noexcept {
    EINSUMS_INTERNAL_METADATA_ADD(domain, *id, key, __itt_metadata_u64, 1, const_cast<std::uint64_t *>(&data));
}

void itt_metadata_add(___itt_domain *domain, ___itt_id *id, ___itt_string_handle *key, double const &data) noexcept {
    EINSUMS_INTERNAL_METADATA_ADD(domain, *id, key, __itt_metadata_double, 1, const_cast<double *>(&data));
}

void itt_metadata_add(___itt_domain *domain, ___itt_id *id, ___itt_string_handle *key, char const *data) noexcept {
    EINSUMS_INTERNAL_METADATA_STR_ADD(domain, *id, key, data);
}

void itt_metadata_add(___itt_domain *domain, ___itt_id *id, ___itt_string_handle *key, void const *data) noexcept {
    EINSUMS_INTERNAL_METADATA_ADD(domain, *id, key, __itt_metadata_unknown, 1, const_cast<void *>(data));
}

#endif // EINSUMS_HAVE_ITTNOTIFY
