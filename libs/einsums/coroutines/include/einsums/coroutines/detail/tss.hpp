//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/assert.hpp>

#include <cstddef>
#include <map>
#include <memory>
#include <utility>

namespace einsums::threads::coroutines::detail {

struct tss_storage;

#if defined(EINSUMS_HAVE_THREAD_LOCAL_STORAGE)
//////////////////////////////////////////////////////////////////////////
struct tss_cleanup_function {
    virtual ~tss_cleanup_function() {}

    virtual void operator()(void *data) = 0;
};

//////////////////////////////////////////////////////////////////////////
struct tss_data_node {
  private:
    std::shared_ptr<tss_cleanup_function> _func;
    void                                 *_value;

  public:
    tss_data_node() : _value(nullptr) {}

    tss_data_node(void *val) : _func(), _value(val) {}

    tss_data_node(std::shared_ptr<tss_cleanup_function> f, void *val) : _func(std::move(f)), _value(val) {}

    tss_data_node(tss_data_node const &)                     = delete;
    auto operator=(tss_data_node const &) -> tss_data_node & = delete;

    tss_data_node(tss_data_node &&other) noexcept : _func(EINSUMS_MOVE(other._func)), _value(other._value) {
        other._value = nullptr;
    }

    auto operator=(tss_data_node &&other) noexcept -> tss_data_node & {
        cleanup();
        _func        = EINSUMS_MOVE(other._func);
        _value       = other._value;
        other._value = nullptr;
        return *this;
    }

    ~tss_data_node() { cleanup(); }

    template <typename T>
    auto get_data() const -> T {
        EINSUMS_ASSERT(value_ != nullptr);
        return *reinterpret_cast<T *>(_value);
    }

    template <typename T>
    void set_data(T const &val) {
        if (_value == nullptr)
            _value = new T(val);
        else
            *reinterpret_cast<T *>(_value) = val;
    }

    void cleanup(bool cleanup_existing = true);

    void reinit(std::shared_ptr<tss_cleanup_function> const &f, void *data, bool cleanup_existing) {
        cleanup(cleanup_existing);
        _func  = f;
        _value = data;
    }

    [[nodiscard]] auto get_value() const -> void * { return _value; }
};

//////////////////////////////////////////////////////////////////////////
class tss_storage {
  private:
    using tss_node_data_map = std::map<void const *, tss_data_node>;

    auto find_entry(void const *key) const -> tss_data_node const * {
        auto it = _data.find(key);
        if (it == _data.end())
            return nullptr;
        return &(it->second);
    }

    auto find_entry(void const *key) -> tss_data_node * {
        auto it = _data.find(key);
        if (it == _data.end())
            return nullptr;
        return &(it->second);
    }

  public:
    tss_storage() = default;

    ~tss_storage() {}

    [[nodiscard]] auto get_thread_data() const -> std::size_t { return 0; }
    auto               set_thread_data(std::size_t /*val*/) -> std::size_t { return 0; }

    auto find(void const *key) -> tss_data_node * {
        auto current_node = _data.find(key);
        if (current_node != _data.end())
            return &current_node->second;
        return nullptr;
    }

    void insert(void const *key, std::shared_ptr<tss_cleanup_function> const &func, void *tss_data) {
        _data.insert(std::make_pair(key, tss_data_node(func, tss_data)));
    }

    void insert(void const *key, void *tss_data) {
        std::shared_ptr<tss_cleanup_function> func;
        insert(key, func, tss_data); //-V614
    }

    void erase(void const *key, bool cleanup_existing) {
        tss_data_node *node = find(key);
        if (node) {
            if (!cleanup_existing)
                node->cleanup(false);
            _data.erase(key);
        }
    }

  private:
    tss_node_data_map _data;
};

//////////////////////////////////////////////////////////////////////////
EINSUMS_EXPORT auto find_tss_data(void const *key) -> tss_data_node *;
EINSUMS_EXPORT auto get_tss_data(void const *key) -> void *;
EINSUMS_EXPORT void add_new_tss_node(void const *key, std::shared_ptr<tss_cleanup_function> const &func,
                                     void *tss_data);
EINSUMS_EXPORT void erase_tss_node(void const *key, bool cleanup_existing = false);
EINSUMS_EXPORT void set_tss_data(void const *key, std::shared_ptr<tss_cleanup_function> const &func,
                                 void *tss_data = nullptr, bool cleanup_existing = false);

//////////////////////////////////////////////////////////////////////////
EINSUMS_EXPORT auto create_tss_storage() -> tss_storage *;
EINSUMS_EXPORT void delete_tss_storage(tss_storage *&storage);

EINSUMS_EXPORT auto get_tss_thread_data(tss_storage *storage) -> std::size_t;
EINSUMS_EXPORT auto set_tss_thread_data(tss_storage *storage, std::size_t) -> std::size_t;
#endif

} // namespace einsums::threads::coroutines::detail