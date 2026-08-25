-- ============================================================
-- MANORA - Core Database Schema (PostgreSQL / Supabase)
-- Digital Mental Health & Psychological Support System for Students
-- ============================================================

-- Enable UUID extension if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ------------------------------------------------------------
-- 1. USERS
-- Student profiles and system identifiers.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE,
    full_name VARCHAR(255),
    academic_program VARCHAR(255),
    year_of_study INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ------------------------------------------------------------
-- 2. SESSIONS
-- Interaction sessions between student and Buddy.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ------------------------------------------------------------
-- 3. GOALS
-- Student goals (e.g. academic progress, placement prep, sleep).
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS goals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    target_date DATE,
    status VARCHAR(50) NOT NULL DEFAULT 'active'
        CHECK (status IN ('active', 'completed', 'paused', 'abandoned')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ------------------------------------------------------------
-- 4. INTERACTIONS
-- Stores conversation messages (user / buddy).
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS interactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'buddy')),
    raw_text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ------------------------------------------------------------
-- 5. INTERACTION ANALYSES
-- Structured output of the Emotion Agent (stored as JSONB).
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS interaction_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    interaction_id UUID NOT NULL REFERENCES interactions(id) ON DELETE CASCADE,
    analysis_version INTEGER NOT NULL DEFAULT 1,
    analysis_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (interaction_id, analysis_version)
);

-- ------------------------------------------------------------
-- 6. CANDIDATE MEMORIES
-- Extracted candidate memories for Data Agent / Memory Engine.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS candidate_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    interaction_id UUID NOT NULL REFERENCES interactions(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    context_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    emotional_state_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    importance NUMERIC(4,3) CHECK (importance >= 0 AND importance <= 1),
    confidence NUMERIC(4,3) CHECK (confidence >= 0 AND confidence <= 1),
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'accepted', 'rejected', 'merged')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ------------------------------------------------------------
-- 7. BUDDY STATES
-- Current internal emotional state of Buddy per student.
-- All numeric values are strictly bounded between 0.0 and 1.0.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS buddy_states (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    happiness NUMERIC(4,3) NOT NULL CHECK (happiness >= 0 AND happiness <= 1),
    sadness NUMERIC(4,3) NOT NULL CHECK (sadness >= 0 AND sadness <= 1),
    frustration NUMERIC(4,3) NOT NULL CHECK (frustration >= 0 AND frustration <= 1),
    concern NUMERIC(4,3) NOT NULL CHECK (concern >= 0 AND concern <= 1),
    warmth NUMERIC(4,3) NOT NULL CHECK (warmth >= 0 AND warmth <= 1),
    patience NUMERIC(4,3) NOT NULL CHECK (patience >= 0 AND patience <= 1),
    energy NUMERIC(4,3) NOT NULL CHECK (energy >= 0 AND energy <= 1),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ------------------------------------------------------------
-- 8. BUDDY STATE HISTORY
-- Audit trail of Buddy state transitions over time.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS buddy_state_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    previous_state JSONB NOT NULL,
    new_state JSONB NOT NULL,
    trigger_interaction_id UUID REFERENCES interactions(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- INDEXES
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_goals_user_id ON goals(user_id);
CREATE INDEX IF NOT EXISTS idx_interactions_user_id ON interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_interactions_session_id ON interactions(session_id);
CREATE INDEX IF NOT EXISTS idx_interactions_created_at ON interactions(created_at);
CREATE INDEX IF NOT EXISTS idx_interaction_analyses_interaction_id ON interaction_analyses(interaction_id);
CREATE INDEX IF NOT EXISTS idx_candidate_memories_user_id ON candidate_memories(user_id);
CREATE INDEX IF NOT EXISTS idx_candidate_memories_status ON candidate_memories(status);
CREATE INDEX IF NOT EXISTS idx_candidate_memories_interaction_id ON candidate_memories(interaction_id);
CREATE INDEX IF NOT EXISTS idx_buddy_states_user_id ON buddy_states(user_id);
CREATE INDEX IF NOT EXISTS idx_buddy_state_history_user_id ON buddy_state_history(user_id);

-- ------------------------------------------------------------
-- 9. LONG TERM MEMORIES
-- Stores long term memories.
-- ------------------------------------------------------------


CREATE TABLE long_term_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    user_id UUID NOT NULL,

    content TEXT NOT NULL,

    importance REAL NOT NULL DEFAULT 0.5
        CHECK (importance >= 0 AND importance <= 1),

    confidence REAL NOT NULL DEFAULT 0.5
        CHECK (confidence >= 0 AND confidence <= 1),

    emotions JSONB NOT NULL DEFAULT '[]'::jsonb,

    evidence_ids JSONB NOT NULL DEFAULT '[]'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE INDEX idx_long_term_memories_user_id
    ON long_term_memories(user_id);

CREATE INDEX idx_long_term_memories_active
    ON long_term_memories(user_id, is_active);