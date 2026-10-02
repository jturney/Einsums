-- Migration 004: Record how each result was collected.
-- 'profiler' (a structured benchmark_result event from the profiler server), 'stdout' (the
-- test's printed line, when no server reported it), or 'build' (the build-time row).
-- Rows from before this column existed came from stdout.
ALTER TABLE results ADD COLUMN source TEXT;
UPDATE results SET source = CASE WHEN test_binary = 'build' THEN 'build' ELSE 'stdout' END WHERE source IS NULL;
