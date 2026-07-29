# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

from docutils import nodes


def setup(app):
    app.add_role('einsums-issue', autolink('https://github.com/Einsums/Einsums/issues/%s', "Issue #"))
    app.add_role('einsums-pr', autolink('https://github.com/Einsums/Einsums/pull/%s', "PR #"))
    app.add_role('cppreference-header', autolink('http://en.cppreference.com/w/cpp/header/%s'))
    app.add_role('cppreference-algorithm', autolink('http://en.cppreference.com/w/cpp/algorithm/%s'))
    app.add_role('cppreference-memory', autolink('http://en.cppreference.com/w/cpp/memory/%s'))
    app.add_role('cppreference-container', autolink('http://en.cppreference.com/w/cpp/container/%s'))
    app.add_role('cppreference-generic', autolink_generic('http://en.cppreference.com/w/cpp/%s/%s'))


def autolink(pattern, prefix=''):
    def role(name, rawtext, text, lineno, inliner, options={}, content=[]):
        url = pattern % (text,)
        node = nodes.reference(rawtext, prefix + text, refuri=url, **options)
        return [node], []

    return role


# The text in the rst file should be:
# :cppreference-generic:`base_path,typename[,shown]`, for instance `thread,barrier`
def autolink_generic(pattern):
    def role(name, rawtext, text, lineno, inliner, options=None, content=None):
        if options is None :
            options = {}
        if content is None :
            content = []
        text_parts = [p.strip() for p in text.split(',')]
        shown_text = None
        if len(text_parts) >= 3:
            shown_text = text_parts[2]
            url = pattern % (text_parts[0], text_parts[1])
        elif len(text_parts) == 2:
            shown_text = text_parts[1]
            url = pattern % (text_parts[0], text_parts[1])
        else:
            shown_text = text_parts[0]
            url = pattern % (text_parts[0], text_parts[0])
        node = nodes.reference(rawtext, "std::" + shown_text, refuri=url, **options)
        return [node], []

    return role
