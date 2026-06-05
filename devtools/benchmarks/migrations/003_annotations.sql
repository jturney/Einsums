-- Migration 003: Add annotations column to results for structured benchmark metadata.
ALTER TABLE results ADD COLUMN annotations TEXT;
