-- ============================================================
-- MANORA - Emotion Agent Storage
-- ============================================================

-- ------------------------------------------------------------
-- 1. INTERACTIONS
-- Stores the actual conversation messages.
-- role = user / buddy
-- ------------------------------------------------------------

CREATE TABLE interactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    user_id UUID NOT NULL,
    session_id UUID NOT NULL,

    role VARCHAR(20) NOT NULL
        CHECK (role IN ('user', 'buddy')),

    raw_text TEXT NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ------------------------------------------------------------
-- 2. INTERACTION ANALYSES
-- Stores the complete structured output of the Emotion Agent.
-- The entire agent response is stored as JSONB.
-- ------------------------------------------------------------

CREATE TABLE interaction_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    interaction_id UUID NOT NULL
        REFERENCES interactions(id)
        ON DELETE CASCADE,

    analysis_version INTEGER NOT NULL DEFAULT 1,

    analysis_json JSONB NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (interaction_id, analysis_version)
);


-- ------------------------------------------------------------
-- 3. CANDIDATE MEMORIES
-- Stores memories identified by the Emotion Agent
-- for later processing by the Data Agent.
-- ------------------------------------------------------------

CREATE TABLE candidate_memories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    interaction_id UUID NOT NULL
        REFERENCES interactions(id)
        ON DELETE CASCADE,

    user_id UUID NOT NULL,

    content TEXT NOT NULL,

    context_json JSONB NOT NULL DEFAULT '{}'::jsonb,

    emotional_state_json JSONB NOT NULL DEFAULT '{}'::jsonb,

    importance NUMERIC(4,3)
        CHECK (importance >= 0 AND importance <= 1),

    confidence NUMERIC(4,3)
        CHECK (confidence >= 0 AND confidence <= 1),

    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (
            status IN (
                'pending',
                'accepted',
                'rejected',
                'merged'
            )
        ),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ============================================================
-- INDEXES
-- ============================================================

CREATE INDEX idx_interactions_user_id
    ON interactions(user_id);

CREATE INDEX idx_interactions_session_id
    ON interactions(session_id);

CREATE INDEX idx_interactions_created_at
    ON interactions(created_at);

CREATE INDEX idx_interaction_analyses_interaction_id
    ON interaction_analyses(interaction_id);

CREATE INDEX idx_candidate_memories_user_id
    ON candidate_memories(user_id);

CREATE INDEX idx_candidate_memories_status
    ON candidate_memories(status);

CREATE INDEX idx_candidate_memories_interaction_id
    ON candidate_memories(interaction_id);