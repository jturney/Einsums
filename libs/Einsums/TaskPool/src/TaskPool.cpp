//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Logging.hpp>
#include <Einsums/Runtime/ShutdownFunction.hpp>
#include <Einsums/TaskPool/TaskPool.hpp>

#if defined(EINSUMS_HAVE_PROFILER)
#    include <Einsums/Profile/Profile.hpp>
#endif

#include <chrono>
#include <random>
#include <thread>
#include <utility>

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace einsums::task_pool {

EINSUMS_SINGLETON_IMPL(TaskPool)

void wake_pool_workers() {
    try {
        TaskPool::get_singleton().wake_workers();
    } catch (...) { // NOLINT
    }
}

// Thread-local worker ID (-1 for non-worker threads)
namespace {
thread_local int tls_worker_id = -1;
}

TaskPool::TaskPool() {
    // Determine thread count
    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 4;

#ifdef _OPENMP
    // Match OpenMP thread count if set
    int const omp_threads = omp_get_max_threads();
    if (omp_threads > 0) {
        num_threads = static_cast<size_t>(omp_threads);
    }
#endif

    // Reserve at least 1 worker
    if (num_threads < 1)
        num_threads = 1;

    EINSUMS_LOG_INFO("TaskPool: creating {} worker threads", num_threads);

    _workers.reserve(num_threads);
    for (size_t i = 0; i < num_threads; i++) {
        auto worker               = std::make_unique<Worker>();
        worker->context.worker_id = static_cast<int>(i);
        _workers.push_back(std::move(worker));
    }

    // Worker threads are started lazily on first enqueue() call,
    // not here. This avoids creating threads during static initialization
    // or program startup when no tasks may ever be submitted.

    // Note: shutdown function is registered lazily in enqueue() when workers
    // are actually started, to ensure workers exist when shutdown() is called.

    // Register profiler server handler for metrics
#if defined(EINSUMS_HAVE_PROFILER)
    auto *srv = profile::Profiler::instance().server();
    if (srv) {
        srv->register_handler("get_taskpool_metrics", [this](std::string const &) {
            auto        m    = snapshot_metrics();
            std::string json = "{\"total_submitted\":" + std::to_string(m.total_submitted);
            json += ",\"total_completed\":" + std::to_string(m.total_completed);
            json += ",\"total_steals\":" + std::to_string(m.total_steals);
            json += ",\"active_workers\":" + std::to_string(m.active_workers);
            json += ",\"num_workers\":" + std::to_string(num_workers());
            json += ",\"per_worker_executed\":[";
            for (size_t i = 0; i < m.per_worker_executed.size(); i++) {
                if (i > 0)
                    json += ",";
                json += std::to_string(m.per_worker_executed[i]);
            }
            json += "],\"per_worker_stolen\":[";
            for (size_t i = 0; i < m.per_worker_stolen.size(); i++) {
                if (i > 0)
                    json += ",";
                json += std::to_string(m.per_worker_stolen[i]);
            }
            json += "]}";
            return json;
        });
    }
#endif
}

TaskPool::~TaskPool() {
    shutdown();
}

void TaskPool::shutdown() {
    if (_shutdown_flag.exchange(true)) {
        return; // Already shutting down
    }

    // Wake all parked workers repeatedly so they see the shutdown flag.
    for (int i = 0; i < 10; i++) {
        _notify_cv.notify_all();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // Join all workers. Workers check shutdown_flag_ every 10ms (park timeout)
    // so this should complete within ~20ms.
    for (auto &w : _workers) {
        if (w->thread.joinable()) {
            w->thread.join();
        }
    }
}

void TaskPool::worker_loop(size_t worker_id) {
    tls_worker_id = static_cast<int>(worker_id);

#ifdef _OPENMP
    // Run BLAS (and any other OpenMP code) single-threaded on worker threads.
    // omp_set_num_threads sets the thread-scoped nthreads-var ICV, so this
    // affects only parallel regions this worker encounters; the main thread
    // and its OpenMP parallelism are untouched. This is both the right policy,
    // since the pool parallelizes across nodes so nested per-node BLAS threads
    // only oversubscribe, and a correctness fix: a libomp parallel region opened
    // from a foreign (non-main) thread, with several workers doing so concurrently,
    // can deadlock libomp's thread pool. For example, a control-flow node whose body
    // runs inline on a worker and calls multithreaded BLAS would intermittently
    // wedge at the fork/join barrier.
    omp_set_num_threads(1);
#endif

    // Register thread name with profiler (safe to call from any thread after
    // Profiler singleton is initialized; the thread-local ring buffer is
    // created lazily on first push, and set_thread_name just records the name).
#if defined(EINSUMS_HAVE_PROFILER)
    try {
        profile::Profiler::instance().set_thread_name(fmt::format("taskpool-worker-{}", worker_id));
    } catch (...) { // NOLINT
        // Profiler may not be initialized yet during early startup
    }
#endif

    auto        &my_deque = _workers[worker_id]->deque;
    std::mt19937 rng(static_cast<unsigned>(worker_id * 2654435761ULL));
    size_t const nw = _workers.size();

    constexpr int spin_count = 64;

    while (!_shutdown_flag.load(std::memory_order_relaxed)) {
        // 1. Try own deque
        auto task = my_deque.pop();
        if (task) {
            (*task)();
            _workers[worker_id]->tasks_executed.fetch_add(1, std::memory_order_relaxed);
            _total_completed.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        // 2. Check external queue (tasks from non-worker threads)
        {
            std::scoped_lock const lock(_external_mutex);
            if (!_external_queue.empty()) {
                task = std::move(_external_queue.front());
                _external_queue.pop();
            }
        }
        if (task) {
            (*task)();
            _workers[worker_id]->tasks_executed.fetch_add(1, std::memory_order_relaxed);
            _total_completed.fetch_add(1, std::memory_order_relaxed);
            continue;
        }

        // 3. Try stealing from other workers
        bool found = false;
        for (int attempt = 0; attempt < spin_count && !_shutdown_flag.load(std::memory_order_relaxed); attempt++) {
            size_t const victim = rng() % nw;
            if (victim == worker_id)
                continue;

            task = _workers[victim]->deque.steal();
            if (task) {
                (*task)();
                _workers[worker_id]->tasks_executed.fetch_add(1, std::memory_order_relaxed);
                _workers[worker_id]->tasks_stolen.fetch_add(1, std::memory_order_relaxed);
                _total_completed.fetch_add(1, std::memory_order_relaxed);
                found = true;
                break;
            }
        }
        if (found)
            continue;

        // 4. Park with condition variable
        _parked_count.fetch_add(1, std::memory_order_relaxed);
        {
            std::unique_lock lock(_notify_mutex);
            _notify_cv.wait_for(lock, std::chrono::milliseconds(1));
        }
        _parked_count.fetch_sub(1, std::memory_order_relaxed);
    }
}

void TaskPool::enqueue(std::function<void()> task) {
    // Lazy start: create worker threads on first enqueue.
    // Also register shutdown function so workers are joined during
    // einsums::finalize(), BEFORE the OMP parallel region's barrier.
    // Without this, OMP waits for worker threads that called OMP-aware
    // functions (like BLAS) and will never arrive at the barrier.
    static std::once_flag start_flag;
    std::call_once(start_flag, [this]() {
        for (size_t i = 0; i < _workers.size(); i++) {
            _workers[i]->thread = std::thread(&TaskPool::worker_loop, this, i);
        }
        try {
            einsums::register_pre_shutdown_function([this]() { shutdown(); });
        } catch (...) { // NOLINT
        }
    });

    _total_submitted.fetch_add(1, std::memory_order_relaxed);

    // If called from a worker thread, push to own deque (safe: SPMC owner push).
    if (tls_worker_id >= 0 && std::cmp_less(tls_worker_id, _workers.size())) {
        _workers[static_cast<size_t>(tls_worker_id)]->deque.push(std::move(task));
    } else {
        // External submission: push to shared MPMC queue (mutex-protected).
        // Cannot use worker deques from external threads; they are SPMC
        // (single-producer only from the owner thread).
        std::scoped_lock const lock(_external_mutex);
        _external_queue.push(std::move(task));
    }

    // Wake one parked worker for the one new task. notify_all here woke
    // every parked worker on every push - including a worker's own-deque
    // push during dataflow execution - so an n-node graph triggered n
    // thundering herds. A lost or insufficient wakeup self-heals: workers
    // park with a 1ms timeout and re-scan all queues on wake.
    if (_parked_count.load(std::memory_order_relaxed) > 0) {
        _notify_cv.notify_one();
    }
}

void TaskPool::help_until(std::function<bool()> const &predicate) {
    // Calling thread participates in work-stealing while waiting.
    // Also wakes parked workers to ensure all enqueued work gets processed.
    std::mt19937 rng(42); // NOLINT(bugprone-random-generator-seed)
    size_t const nw = _workers.size();

    while (!predicate() && !_shutdown_flag.load(std::memory_order_relaxed)) {
        // Wake a parked worker (same reasoning as enqueue: this loop runs
        // every ~50us, so notify_all amounted to a periodic herd wakeup;
        // the 1ms park timeout backstops any missed signal).
        if (_parked_count.load(std::memory_order_relaxed) > 0) {
            _notify_cv.notify_one();
        }

        // Try to steal work ourselves
        bool found = false;
        for (size_t attempt = 0; attempt < nw; attempt++) {
            size_t const victim = rng() % nw;
            auto         task   = _workers[victim]->deque.steal();
            if (task) {
                (*task)();
                _total_completed.fetch_add(1, std::memory_order_relaxed);
                found = true;
                break;
            }
        }
        if (!found) {
            // Nothing to steal; workers are processing. Short sleep to avoid spinning.
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }
}

bool TaskPool::is_worker_thread() const {
    return tls_worker_id >= 0 && std::cmp_less(tls_worker_id, _workers.size());
}

int TaskPool::current_worker_id() {
    return tls_worker_id;
}

TaskPool::Metrics TaskPool::snapshot_metrics() const {
    Metrics m;
    m.total_submitted = _total_submitted.load();
    m.total_completed = _total_completed.load();

    size_t const nw = _workers.size();
    m.per_worker_executed.resize(nw);
    m.per_worker_stolen.resize(nw);
    m.active_workers = 0;
    m.total_steals   = 0;

    for (size_t i = 0; i < nw; i++) {
        m.per_worker_executed[i] = _workers[i]->tasks_executed.load();
        m.per_worker_stolen[i]   = _workers[i]->tasks_stolen.load();
        m.total_steals += m.per_worker_stolen[i];
        if (!_workers[i]->deque.empty_approx()) {
            m.active_workers++;
        }
    }

    return m;
}

TaskGroup TaskPool::submit_group(std::string name, std::vector<std::function<void()>> tasks) {
    TaskGroup group(std::move(name), tasks.size());

    for (auto &task : tasks) {
        auto grp_remaining = group._remaining;
        auto grp_mutex     = group._mutex;
        auto grp_cv        = group._cv;

        enqueue([t = std::move(task), remaining = std::move(grp_remaining), mtx = std::move(grp_mutex), cv = std::move(grp_cv)]() {
            t();
            if (remaining->fetch_sub(1) == 1) {
                {
                    std::scoped_lock const lock(*mtx);
                }
                cv->notify_all();
            }
        });
    }

    return group;
}

} // namespace einsums::task_pool
