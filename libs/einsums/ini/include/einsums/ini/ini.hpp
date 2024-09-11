//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/string_util/to_string.hpp>

#include <functional>
#include <iosfwd>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// suppress warnings about dependent classes not being exported from the dll
#if defined(EINSUMS_MSVC_WARNING_PRAGMA)
#    pragma warning(push)
#    pragma warning(disable : 4091 4251 4231 4275 4660)
#endif

namespace einsums::detail {
class EINSUMS_EXPORT section {
  public:
    using entry_changed_func = std::function<void(const std::string &, const std::string &)>;
    using entry_type         = std::pair<std::string, entry_changed_func>;
    using entry_map          = std::map<std::string, entry_type>;
    using section_map        = std::map<std::string, section>;

  private:
    section *this_() { return this; }

    using mutex_type = std::mutex;

    section    *_root;
    entry_map   _entries;
    section_map _sections;
    std::string _name;
    std::string _parent_name;

    mutable mutex_type _mutex;

  protected:
    void line_msg(std::string msg, std::string const &file, int lnum = 0, std::string const &line = "");

    section &clone_from(section const &rhs, section *root = nullptr);

  private:
    void add_section(std::unique_lock<mutex_type> &l, std::string const &sec_name, section &sec, section *root = nullptr);
    bool has_section(std::unique_lock<mutex_type> &l, std::string const &sec_name) const;

    section       *get_section(std::unique_lock<mutex_type> &l, std::string const &sec_name);
    section const *get_section(std::unique_lock<mutex_type> &l, std::string const &sec_name) const;

    ///////////////////////////////////////////////////////////////////////////
    section *add_section_if_new(std::unique_lock<mutex_type> &l, std::string const &sec_name);

    void add_entry(std::unique_lock<mutex_type> &l, std::string const &fullkey, std::string const &key, std::string val);
    void add_entry(std::unique_lock<mutex_type> &l, std::string const &fullkey, std::string const &key, entry_type const &val);

    bool        has_entry(std::unique_lock<mutex_type> &l, std::string const &key) const;
    std::string get_entry(std::unique_lock<mutex_type> &l, std::string const &key) const;
    std::string get_entry(std::unique_lock<mutex_type> &l, std::string const &key, std::string const &dflt) const;

    void add_notification_callback(std::unique_lock<mutex_type> &l, std::string const &key, entry_changed_func const &callback);

  public:
    section();
    explicit section(std::string const &filename, section *root = nullptr);
    section(section const &in);
    ~section() = default;

    section &operator=(section const &rhs);

    void parse(std::string const &sourcename, std::vector<std::string> const &lines, bool verify_existing = true,
               bool weed_out_comments = true, bool replace_existing = true);
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    void parse(std::string const &sourcename, std::string const &line, bool verify_existing = true, bool weed_out_comments = true,
               bool replace_existing = true);
    // NOLINTEND(bugprone-easily-swappable-parameters)

    void read(std::string const &filename);
    void merge(std::string const &second);
    void merge(section &second);
    void dump(int ind = 0) const;
    void dump(int ind, std::ostream &strm) const;

    void               add_section(std::string const &sec_name, section &sec, section *root = nullptr);
    section           *add_section_if_new(std::string const &sec_name);
    bool               has_section(std::string const &sec_name) const;
    section           *get_section(std::string const &sec_name);
    section const     *get_section(std::string const &sec_name) const;
    section_map       &get_sections() noexcept;
    section_map const &get_sections() const noexcept;
    void               add_entry(std::string const &key, entry_type const &val);
    void               add_entry(std::string const &key, std::string const &val);
    bool               has_entry(std::string const &key) const;
    std::string        get_entry(std::string const &key) const;
    std::string        get_entry(std::string const &key, std::string const &dflt) const;
    void               add_notification_callback(std::string const &key, entry_changed_func const &callback);
    entry_map const   &get_entries() const noexcept;

    template <typename T>
    std::string get_entry(std::string const &key, T dflt) const {
        std::unique_lock<mutex_type> l(_mutex);
        return get_entry(l, key, einsums::string_util::to_string(dflt));
    }

  private:
    std::string expand(std::unique_lock<mutex_type> &l, std::string in) const;

    void expand(std::unique_lock<mutex_type> &l, std::string &, std::string::size_type) const;
    void expand_bracket(std::unique_lock<mutex_type> &l, std::string &, std::string::size_type) const;
    void expand_brace(std::unique_lock<mutex_type> &l, std::string &, std::string::size_type) const;

    std::string expand_only(std::unique_lock<mutex_type> &l, std::string in, std::string const &expand_this) const;

    void expand_only(std::unique_lock<mutex_type> &l, std::string &, std::string::size_type, std::string const &expand_this) const;
    void expand_bracket_only(std::unique_lock<mutex_type> &l, std::string &, std::string::size_type, std::string const &expand_this) const;
    void expand_brace_only(std::unique_lock<mutex_type> &l, std::string &, std::string::size_type, std::string const &expand_this) const;

  public:
    std::string expand(std::string const &str) const;
    void        expand(std::string &str, std::string::size_type len) const;
    void        set_root(section *r, bool recursive = false);
    section    *get_root() const noexcept;
    std::string get_name() const;
    std::string get_parent_name() const;
    std::string get_full_name() const;
    void        set_name(std::string const &name);
};

} // namespace einsums::detail
