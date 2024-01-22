//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>
#if defined(EINSUMS_HAVE_THREAD_LOCAL_STORAGE)
#    include <einsums/assert.hpp>
#    include <einsums/coroutines/coroutine.hpp>
#    include <einsums/coroutines/detail/coroutine_self.hpp>
#    include <einsums/coroutines/detail/tss.hpp>
#    include <einsums/modules/errors.hpp>
#    include <einsums/type_support/unused.hpp>

#    include <cstddef>
#    include <map>
#    include <memory>

namespace einsums::threads::coroutines::detail {
///////////////////////////////////////////////////////////////////////////
void tss_data_node::cleanup(bool cleanup_existing) {
#    ifdef EINSUMS_HAVE_THREAD_LOCAL_STORAGE
    if (cleanup_existing && _func && (_value != nullptr)) {
        (*_func)(_value);
    }
    _func.reset();
    _value = nullptr;
#    else
    EINSUMS_UNUSED(cleanup_existing);
#    endif
}

///////////////////////////////////////////////////////////////////////////
tss_storage *create_tss_storage() {
#    ifdef EINSUMS_HAVE_THREAD_LOCAL_STORAGE
    return new tss_storage;
#    else
    throw std::runtime_error("thread local storage has been disabled at configuration time, "
                             "please specify EINSUMS_WITH_THREAD_LOCAL_STORAGE=ON to cmake");
    return nullptr;
#    endif
}

void delete_tss_storage(tss_storage *&storage) {
#    ifdef EINSUMS_HAVE_THREAD_LOCAL_STORAGE
    delete storage;
    storage = nullptr;
#    else
    EINSUMS_UNUSED(storage);
#    endif
}

std::size_t get_tss_thread_data(tss_storage *) {
#    ifdef EINSUMS_HAVE_THREAD_LOCAL_STORAGE
    coroutine_self *self = coroutine_self::get_self();
    if (nullptr == self) {
        EINSUMS_THROW_EXCEPTION(einsums::error::null_thread_id,
                                "einsums::threads::coroutines::detail::get_tss_thread_data",
                                "null thread id encountered");
        return 0;
    }

    tss_storage *tss_map = self->get_thread_tss_data();
    if (nullptr == tss_map)
        return 0;

    tss_data_node *node = tss_map->find(nullptr);
    if (nullptr == node)
        return 0;

    return node->get_data<std::size_t>();
#    else
    throw std::runtime_error("thread local storage has been disabled at configuration time, "
                             "please specify EINSUMS_WITH_THREAD_LOCAL_STORAGE=ON to cmake");
    return 0;
#    endif
}

std::size_t set_tss_thread_data(tss_storage *, std::size_t data) {
#    ifdef EINSUMS_HAVE_THREAD_LOCAL_STORAGE
    coroutine_self *self = coroutine_self::get_self();
    if (nullptr == self) {
        EINSUMS_THROW_EXCEPTION(einsums::error::null_thread_id,
                                "einsums::threads::coroutines::detail::set_tss_thread_data",
                                "null thread id encountered");
        return 0;
    }

    tss_storage *tss_map = self->get_or_create_thread_tss_data();
    if (nullptr == tss_map) {
        throw std::bad_alloc();
        return 0;
    }

    tss_data_node *node = tss_map->find(nullptr);
    if (nullptr == node) {
        tss_map->insert(nullptr, new std::size_t(data)); //-V508
        return 0;
    }

    std::size_t prev_val = node->get_data<std::size_t>();
    node->set_data(data);

    return prev_val;
#    else
    EINSUMS_UNUSED(data);
    throw std::runtime_error("thread local storage has been disabled at configuration time, "
                             "please specify EINSUMS_WITH_THREAD_LOCAL_STORAGE=ON to cmake");
    return 0;
#    endif
}

///////////////////////////////////////////////////////////////////////////
tss_data_node *find_tss_data(void const *key) {
#    ifdef EINSUMS_HAVE_THREAD_LOCAL_STORAGE
    coroutine_self *self = coroutine_self::get_self();
    if (nullptr == self) {
        EINSUMS_THROW_EXCEPTION(einsums::error::null_thread_id, "einsums::threads::coroutines::detail::find_tss_data",
                                "null thread id encountered");
        return nullptr;
    }

    tss_storage *tss_map = self->get_thread_tss_data();
    if (nullptr == tss_map)
        return nullptr;

    return tss_map->find(key);
#    else
    EINSUMS_UNUSED(key);
    throw std::runtime_error("thread local storage has been disabled at configuration time, "
                             "please specify EINSUMS_WITH_THREAD_LOCAL_STORAGE=ON to cmake");
    return nullptr;
#    endif
}

void *get_tss_data(void const *key) {
#    ifdef EINSUMS_HAVE_THREAD_LOCAL_STORAGE
    if (tss_data_node *const current_node = find_tss_data(key))
        return current_node->get_value();
#    else
    EINSUMS_UNUSED(key);
#    endif
    return nullptr;
}

void add_new_tss_node(void const *key, std::shared_ptr<tss_cleanup_function> const &func, void *tss_data) {
#    ifdef EINSUMS_HAVE_THREAD_LOCAL_STORAGE
    coroutine_self *self = coroutine_self::get_self();
    if (nullptr == self) {
        EINSUMS_THROW_EXCEPTION(einsums::error::null_thread_id,
                                "einsums::threads::coroutines::detail::add_new_tss_node", "null thread id encountered");
        return;
    }

    tss_storage *tss_map = self->get_or_create_thread_tss_data();
    if (nullptr == tss_map) {
        throw std::bad_alloc();
        return;
    }

    tss_map->insert(key, func, tss_data);
#    else
    EINSUMS_UNUSED(key);
    EINSUMS_UNUSED(func);
    EINSUMS_UNUSED(tss_data);
#    endif
}

void erase_tss_node(void const *key, bool cleanup_existing) {
#    ifdef EINSUMS_HAVE_THREAD_LOCAL_STORAGE
    coroutine_self *self = coroutine_self::get_self();
    if (nullptr == self) {
        EINSUMS_THROW_EXCEPTION(einsums::error::null_thread_id, "einsums::threads::coroutines::detail::erase_tss_node",
                                "null thread id encountered");
        return;
    }

    tss_storage *tss_map = self->get_thread_tss_data();
    if (nullptr != tss_map)
        tss_map->erase(key, cleanup_existing);
#    else
    EINSUMS_UNUSED(key);
    EINSUMS_UNUSED(cleanup_existing);
#    endif
}

void set_tss_data(void const *key, std::shared_ptr<tss_cleanup_function> const &func, void *tss_data,
                  bool cleanup_existing) {
#    ifdef EINSUMS_HAVE_THREAD_LOCAL_STORAGE
    if (tss_data_node *const current_node = find_tss_data(key)) {
        if (func || (tss_data != nullptr))
            current_node->reinit(func, tss_data, cleanup_existing);
        else
            erase_tss_node(key, cleanup_existing);
    } else if (func || (tss_data != nullptr)) {
        add_new_tss_node(key, func, tss_data);
    }
#    else
    EINSUMS_UNUSED(key);
    EINSUMS_UNUSED(func);
    EINSUMS_UNUSED(tss_data);
    EINSUMS_UNUSED(cleanup_existing);
#    endif
}
} // namespace einsums::threads::coroutines::detail
#endif
