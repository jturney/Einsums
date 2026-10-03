/*----------------------------------------------------------------------------------------------
 * Copyright (c) The Einsums Developers. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt in the project root for license information.
 *----------------------------------------------------------------------------------------------*/

/* Waggle from C: the interface header compiles as C, and a C program records zones, annotates
 * them, sets settings, registers a handler and reads its zones back. Exits nonzero on the first
 * failed check. */

#include <Waggle/Waggle.h>

#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(condition)                                                                                                                   \
    do {                                                                                                                                   \
        if (!(condition)) {                                                                                                                \
            fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #condition);                                                  \
            ++failures;                                                                                                                    \
        }                                                                                                                                  \
    } while (0)

static uint32_t site(char const *name, char const *domain) {
    return waggle_register_site(name, strlen(name), __FILE__, __LINE__, "site", waggle_register_domain(domain, strlen(domain)));
}

static int released = 0;

static void release(void *user) {
    (void)user;
    ++released;
}

static void answer(void *user, char const *params, size_t length, waggle_reply *reply) {
    (void)user;
    (void)params;
    (void)length;
    waggle_reply_set(reply, "{}", 2);
}

static waggle_node const *find(waggle_snapshot const *snapshot, char const *path) {
    size_t t;
    for (t = 0; t < waggle_snapshot_thread_count(snapshot); ++t) {
        waggle_node const *node = waggle_snapshot_find(snapshot, t, path, strlen(path));
        if (node != NULL) {
            return node;
        }
    }
    return NULL;
}

int main(void) {
    uint32_t const    outer = site("c: outer", "cdemo");
    uint32_t const    inner = site("c: inner", "cdemo");
    char const *const keys[]   = {"max_distinct_children"};
    char const *const values[] = {"64"};
    char              text[32];
    int               i;

    uint32_t major = 0;
    uint32_t minor = 0;
    waggle_abi_version(&major, &minor);
    CHECK(major == WAGGLE_ABI_MAJOR && minor == WAGGLE_ABI_MINOR);

    waggle_init("cdemo", "1.0", "", "", 0, "");
    waggle_set_enabled(1);
    CHECK(*waggle_enabled_flag() == 1);
    CHECK(outer != 0 && inner != 0 && outer != inner);
    CHECK(site("c: outer", "cdemo") == outer); /* the same description gives the same id */

    /* Settings by name, as text. */
    CHECK(waggle_config_set(keys, values, 1) == 0);
    CHECK(waggle_config_get("max_distinct_children", text, sizeof text) == 2 && strcmp(text, "64") == 0);
    CHECK(waggle_config_get("no such setting", text, sizeof text) == -1);

    /* A handler's user pointer is released once, when it is unregistered. */
    waggle_register_handler("c_method", answer, NULL, release);
    waggle_unregister_handler("c_method");
    CHECK(released == 1);

    for (i = 0; i < 3; ++i) {
        if (waggle_zone_begin(outer, 0)) {
            waggle_annotate_i64(waggle_intern("n", 1), i);
            if (waggle_zone_begin(inner, 0)) {
                waggle_zone_end();
            }
            waggle_zone_end();
        }
    }

    {
        waggle_snapshot *const   snapshot = waggle_snapshot_take(0);
        waggle_node const *const node     = find(snapshot, "c: outer/c: inner");
        waggle_node_stats        stats;
        size_t                   length = 0;
        CHECK(node != NULL);
        if (node != NULL) {
            stats.size = sizeof stats;
            waggle_node_stats_get(node, &stats);
            CHECK(stats.call_count == 3);
            CHECK(strcmp(waggle_node_domain(node, &length), "cdemo") == 0 && length == 5);
        }
        waggle_snapshot_release(snapshot);
    }
    {
        /* Merged across threads, the domain stays. */
        waggle_snapshot *const   snapshot = waggle_snapshot_take(WAGGLE_SNAPSHOT_MERGE_THREADS);
        waggle_node const *const node     = waggle_snapshot_find(snapshot, 0, "c: outer", 8);
        size_t                   length   = 0;
        CHECK(waggle_snapshot_thread_count(snapshot) == 1);
        CHECK(node != NULL && strcmp(waggle_node_domain(node, &length), "cdemo") == 0);
        waggle_snapshot_release(snapshot);
    }

    waggle_finalize("cdemo");
    if (failures == 0) {
        printf("all C interface checks passed\n");
    }
    return failures == 0 ? 0 : 1;
}
