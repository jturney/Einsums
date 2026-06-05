-- Migration 002: Add HPTT selection method to runs.
ALTER TABLE runs ADD COLUMN hptt_method TEXT DEFAULT 'estimate';
