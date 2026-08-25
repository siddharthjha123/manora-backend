Manora — Complete Project Context
1. What is Manora?

Manora is a Digital Mental Health and Psychological Support System for Students in Higher Education.

The core idea is to create a persistent AI companion called Buddy that gradually understands a student's emotional state, goals, behaviours, decisions and recurring patterns.

Unlike a conventional chatbot, Manora does not treat every conversation as an isolated interaction.

The system maintains context over time and uses that context to help Buddy respond appropriately to the student's current situation.

For example, if a student repeatedly says:

"I'll study at 10."

but repeatedly ends up watching a series instead, Manora can identify this behavioural pattern.

Later, instead of simply saying:

"That's okay, try again tomorrow."

Buddy can reason using the student's history:

"You're repeating the same pattern again. Do you actually want to achieve this goal?"

The exact response is generated dynamically based on the student's current context, emotional state, goals, behaviour and Buddy's own internal state.

2. The Core Problem We Are Solving

Higher-education students often experience combinations of:

academic pressure
career uncertainty
placement anxiety
procrastination
inconsistent routines
social problems
stress
frustration
lack of motivation
difficulty maintaining long-term goals
repeated decisions that negatively affect their goals

The problem is not necessarily that students don't know what they should do.

Often, they already know.

The problem is the gap between:

What I want to achieve
        ↓
What I decide to do
        ↓
What I actually do
        ↓
How I feel about it
        ↓
What I do next

Manora attempts to understand this cycle.

It does not simply provide information.

It tries to understand the relationship between emotions, decisions, behaviour, goals and consequences.

3. The Buddy

Buddy is the central user-facing intelligence of Manora.

The Buddy has:

a name chosen by the student
facial expressions/animations
an internal emotional state
a personality
contextual awareness
memory of meaningful historical information
the ability to challenge the student when appropriate

The Buddy communicates with the student primarily through text in V1.

Voice interaction and text-to-speech can be added later.

The Buddy's emotional state is separate from the student's emotional state.

For example:

Student emotion:


frustration = 0.86
guilt       = 0.72
stress      = 0.61

does NOT mean:

Buddy:


frustration = 0.86
guilt       = 0.72

Instead, the Buddy's internal state is updated through a dedicated State Engine.

4. The Overall AI Architecture

Manora consists of several specialized components rather than one giant AI.

The main components are:

1. Interaction Service
2. Memory Retrieval Engine
3. Emotion Agent
4. Emotion ML Model
5. Data Agent
6. Memory Engine
7. Buddy State Engine
8. Buddy Agent

And three storage layers:

PostgreSQL / Supabase
Qdrant
Neo4j

Each has a very specific responsibility.

5. High-Level Architecture
                         STUDENT
                            │
                            │
                            ▼
                  ┌───────────────────┐
                  │ Interaction API   │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Interaction       │
                  │ Service           │
                  └─────────┬─────────┘
                            │
                            ▼
                  ┌───────────────────┐
                  │ Memory Retrieval  │
                  │ Engine            │
                  └─────────┬─────────┘
                            │
                  Is historical context
                       necessary?
                       /          \
                     NO            YES
                     │              │
                     │        ┌─────┴─────┐
                     │        ▼           ▼
                     │     Qdrant       Neo4j
                     │   semantic      relational
                     │    memory         graph
                     │        │           │
                     └────────┴─────┬─────┘
                                    │
                                    ▼
                           Relevant Context
                                    │
                                    ▼
                         ┌────────────────────┐
                         │    Emotion Agent   │
                         │                    │
                         │ ML + LLM reasoning │
                         └─────────┬──────────┘
                                   │
                                   ▼
                           Emotion Analysis
                                   │
                         ┌─────────┴──────────┐
                         │                    │
                         ▼                    ▼
                  Data Agent            State Engine
                         │                    │
                         ▼                    ▼
                  Memory Updates        Buddy State
                         │                    │
                         │                    │
                         ▼                    │
                  Memory Engine               │
                    /       \                 │
                   ▼         ▼                │
              PostgreSQL  Qdrant/Neo4j       │
                                              │
                                              ▼
                                      ┌───────────────┐
                                      │  Buddy Agent  │
                                      └───────┬───────┘
                                              │
                                              ▼
                                      Buddy Response
                                              │
                                              ▼
                                           STUDENT
6. Interaction Service

The Interaction Service is the orchestrator.

The frontend does not need to know how the entire AI pipeline works.

It simply sends:

POST /interactions

with something like:

{
  "user_id": "user_123",
  "session_id": "session_001",
  "text": "I planned to study at 10 but I ended up watching a series for two hours."
}

The Interaction Service coordinates the rest of the system.

It:

receives the interaction
stores it
determines whether historical context is useful
retrieves relevant memories when necessary
invokes the Emotion Agent
processes the resulting emotional analysis
updates Buddy's state
processes memory-related information
invokes the Buddy Agent
returns the final Buddy response

Therefore:

The Interaction Service is the central orchestrator, not an AI agent.

7. Emotion Agent

The Emotion Agent is responsible for understanding the student's current emotional situation.

Its job is NOT simply:

"Detect an emotion."

It performs contextual emotional reasoning.

It considers:

Current message
+
ML emotion signals
+
Recent conversation
+
Relevant memories
+
Goals
+
Behavioural signals
+
Previous context

and produces structured emotion analysis.

For example:

{
  "primary_emotion": "frustration",


  "emotions": [
    {
      "emotion": "frustration",
      "intensity": 0.84,
      "confidence": 0.91
    },
    {
      "emotion": "guilt",
      "intensity": 0.72,
      "confidence": 0.79
    }
  ],


  "behavioral_signals": [
    "avoided planned study activity",
    "continued entertainment despite the planned task"
  ],


  "decision_signals": [
    "chose entertainment instead of studying"
  ],


  "goal_relevance": {
    "related": true,
    "goal": "academic progress"
  }
}

This structured output becomes an important input for the rest of the system.

8. Emotion ML Model

The ML model and Emotion Agent have different responsibilities.

ML model

Answers:

"What emotional signals are present in this text?"

For example:

Input:
"I'm getting really frustrated because I keep doing this."


        ↓


ML model


        ↓


frustration: 0.91
anger:       0.38
sadness:     0.21
stress:      0.67

But these probabilities alone are not enough.

The system then gives those signals to the Emotion Agent, which uses an LLM to reason about the broader context.

Therefore:

ML
↓
Emotion signals


LLM
↓
Contextual emotional reasoning

The ML model is therefore a signal generator, while the Emotion Agent performs the higher-level interpretation.

9. Data Agent

The Data Agent answers a completely different question:

"What information from this interaction is meaningful enough to become part of the student's long-term data?"

Suppose the student says:

"I keep planning to study, but whenever I start watching something, I end up wasting two hours. This has happened several times."

The Data Agent can identify:

Behaviour
Repeated study avoidance
Decision
Choosing entertainment instead of planned study
Event
Missed study session
Emotion
Frustration
Guilt
Goal relationship
Academic / placement goal
Importance
High
Confidence
High

It produces structured candidate memories.

For example:

{
  "content": "Student repeatedly struggles to follow planned study sessions when choosing entertainment.",


  "context": {
    "topic": "study",
    "subtopic": "study_avoidance"
  },


  "emotional_state": [
    {
      "emotion": "frustration",
      "confidence": 0.82
    }
  ],


  "events": [
    {
      "type": "study_avoidance",
      "description": "Student did not follow the planned study session."
    }
  ],


  "behavior": {
    "type": "avoidance"
  },


  "decision": {
    "description": "Student chose entertainment instead of studying."
  },


  "goal_relevance": {
    "related": true,
    "goal": "academic_progress"
  },


  "importance": 0.84,
  "confidence": 0.88
}

So:

Emotion Agent understands the current emotional situation.

Data Agent extracts meaningful long-term information.

They should never be treated as the same component.

10. Memory Engine

The Memory Engine manages how memories are stored and retrieved.

It sits between the AI agents and the databases.

Its responsibility is:

"What should be retrieved?"
"What should be stored?"
"Which memories are relevant?"
"How do we combine semantic and relational context?"

The Memory Engine uses two specialized databases.

11. Qdrant — Semantic Memory

Qdrant is used for semantic similarity.

A meaningful memory such as:

"Student repeatedly struggles to follow planned study sessions when watching entertainment."

is converted into an embedding.

That embedding is stored in Qdrant.

Later the student might say:

"I'm doing the same thing again."

The exact words may be completely different from the original memory.

A normal keyword search may not find the connection.

Qdrant can identify semantic similarity.

Therefore:

Current interaction
        ↓
Embedding
        ↓
Qdrant similarity search
        ↓
Semantically relevant memories

We should not embed every conversation message.

Only meaningful information that should become memory should be represented as long-term semantic memory.

12. Neo4j — Relationship Memory

Qdrant answers:

"What memories are semantically similar?"

Neo4j answers:

"How are these things connected?"

For example:

Student
   │
   ├── HAS_GOAL ───────────────► Placement
   │
   ├── SHOWS_BEHAVIOR ────────► Study Avoidance
   │                                  │
   │                                  ▼
   │                            Entertainment
   │
   ├── EXPERIENCES ────────────► Frustration
   │
   └── MADE_DECISION ──────────► Skip Study
                                      │
                                      ▼
                               Missed Study Session

This becomes extremely important for the future Memory Tree.

The graph can represent:

Memory → Emotion
Memory → Behaviour
Memory → Decision
Decision → Consequence
Behaviour → Goal
Emotion → Event
13. Why Both Qdrant and Neo4j?

They solve different problems.

Technology	Responsibility
PostgreSQL	Canonical application data
Qdrant	Semantic memory retrieval
Neo4j	Relationship/graph reasoning

For example:

Qdrant

"Find memories similar to the student's current situation."

Neo4j

"Show how this decision relates to the student's goals and previous consequences."

This combination allows Manora to eventually understand:

Emotion
   ↓
Decision
   ↓
Behaviour
   ↓
Consequence
   ↓
Goal

which is one of the core ideas behind the Memory Tree.

14. When Do We Retrieve Memory?

We should not query Qdrant and Neo4j on every message.

For example:

"I'm sleepy."

Probably doesn't require a large historical search.

But:

"I'm thinking about giving up on placements again."

should probably retrieve historical context.

Similarly:

"This is happening again just like last month."

strongly suggests that historical information is relevant.

The Memory Retrieval Layer therefore determines whether historical context is useful.

The V1 decision can be lightweight and deterministic, and it can become more sophisticated later.

15. Buddy State Engine

This is a very important component.

Buddy has its own internal state.

Example:

{
  "happiness": 0.52,
  "sadness": 0.10,
  "frustration": 0.44,
  "concern": 0.68,
  "warmth": 0.82,
  "patience": 0.63,
  "energy": 0.55
}

Every value is between:

0 → minimum
1 → maximum

The State Engine updates this state after each interaction.

For example:

Student repeatedly avoids studying
             ↓
Emotion Agent:
frustration = 0.84
guilt       = 0.72
             ↓
State Engine
             ↓
Buddy:
concern ↑
frustration ↑
patience ↓

The State Engine is deterministic, not LLM-controlled.

This is important because we don't want an LLM arbitrarily deciding:

"The student is frustrated, so Buddy should be angry."

Instead:

Emotion Analysis
       ↓
State Rules
       ↓
Buddy State

This makes Buddy's behaviour predictable and controllable.

16. Buddy Agent

The Buddy Agent is responsible for the final response.

It receives:

Current interaction
+
Emotion Analysis
+
Relevant memories
+
Goals
+
Behavioural patterns
+
Decisions
+
Buddy State
+
Recent conversation

It then decides:

what Buddy should say
how Buddy should express it
what type of response it should be
how intense the expression should be

Example:

{
  "text": "You're repeating the same pattern again. Do you actually want to achieve this goal?",


  "expression": "concerned",


  "intensity": 0.72,


  "response_type": "reflection"
}

The frontend can later translate:

expression = concerned
intensity = 0.72

into the Buddy's facial animation.

17. Buddy Does Not Simply Mirror the Student

This is an important design principle.

If:

Student = angry

it does not automatically mean:

Buddy = angry

Instead, Buddy might become:

concerned

or:

supportive

or:

firm

depending on the situation.

For example:

Student:

"I'm so angry at myself. I wasted another three hours."

Buddy might respond:

"I can see why you're frustrated. But you've already noticed the pattern. Let's look at what happened this time."

The Buddy's emotional response is therefore contextual, not simply mirrored.

18. Complete Interaction Example

Suppose the student says:

"I planned to study at 10 AM, but I started watching a series and now it's been two hours. I keep doing this and placements are getting closer."

Step 1 — Interaction Service

Receives and stores the message.

Step 2 — Memory Retrieval

The system recognizes that this is related to:

study
placement
repeated behaviour

Historical context is retrieved.

Qdrant might find:

Student previously struggled with study consistency.

Neo4j might find:

Study avoidance → academic goal → placement
Step 3 — Emotion ML

The ML model detects signals such as:

frustration
guilt
stress
Step 4 — Emotion Agent

Combines:

Current message
+
ML signals
+
historical context
+
goal

and reasons:

Primary emotion: frustration


Secondary emotions:
guilt
stress


Behaviour:
study avoidance


Decision:
continued entertainment


Goal:
placement preparation


Pattern:
repeated behaviour
Step 5 — Data Agent

Extracts meaningful long-term information:

Repeated study avoidance when engaging with entertainment.

and associates it with:

placement goal
Step 6 — Memory Engine

Stores the meaningful information.

PostgreSQL
+
Qdrant
+
Neo4j
Step 7 — Buddy State Engine

Suppose Buddy previously had:

concern = 0.45
patience = 0.72

After this interaction:

concern = 0.68
patience = 0.61
Step 8 — Buddy Agent

Now Buddy sees:

Student is frustrated
+
Student feels guilty
+
This behaviour has happened repeatedly
+
Placement is an important goal
+
Buddy is concerned

The LLM decides that a reflective/challenging response is appropriate.

It produces:

{
  "text": "You're repeating the same pattern again. Do you actually want to achieve this goal?",


  "expression": "concerned",


  "intensity": 0.72,


  "response_type": "reflection"
}
19. What Makes This Different From a Normal Chatbot?

A normal chatbot:

User message
     ↓
LLM
     ↓
Response

Manora:

User
 ↓
Current emotion
 ↓
Historical memory
 ↓
Goals
 ↓
Behaviour
 ↓
Decisions
 ↓
Consequences
 ↓
Student's emotional context
 ↓
Buddy's internal state
 ↓
Reasoning
 ↓
Context-aware response

The central difference is continuity.

Manora is designed to understand:

"What is happening right now?"

but also:

"Has this happened before?"

"What emotion is associated with it?"

"What decision did the student make?"

"What consequence followed?"

"Is this related to one of their goals?"

"How should Buddy respond given its own current state?"

20. Goals Are a Core Part of the System

The student explicitly defines:

Long-term goals

Example:

Get a good placement.

Medium-term goals

Example:

Complete DSA preparation.

Short-term goals

Example:

Study DSA from 10–11 AM today.

Manora can then connect:

Short-term action
       ↓
Medium-term goal
       ↓
Long-term goal

This allows Buddy to understand the consequences of decisions.

For example:

Planned action:
Study at 10 AM


        ↓


Actual behaviour:
Watch series


        ↓


Immediate consequence:
Study session missed


        ↓


Repeated pattern:
Study avoidance


        ↓


Medium-term consequence:
DSA preparation delayed


        ↓


Long-term consequence:
Placement preparation affected

This is the foundation of the future Timeline and Memory Tree features.

21. DASS-21 and Initial Psychological Profiling

During onboarding, Manora will collect information through two stages:

Stage 1 — Manora's own questionnaire

Questions covering areas relevant to the student's:

academic life
social life
goals
habits
motivation
stressors
routines
emotional experiences
Stage 2 — DASS-21

The system will administer the DASS-21 questionnaire as part of the initial assessment process.

The resulting scores become structured information that can be used as context for the system's personalization and emotional reasoning.

It should not be treated as the Buddy independently diagnosing the student.

The assessment data is contextual information, not an excuse for the AI to make clinical diagnoses.

22. The 7-Day Learning Period

The initial experience is designed around a 7-day learning period.

During this period Buddy interacts with the student and gradually learns about:

Goals
+
Emotions
+
Behaviour
+
Decisions
+
Routines
+
Stressors
+
Patterns
+
Consequences

Buddy can ask predefined questions and dynamically adapt questions based on the student's previous responses.

The purpose is not simply to collect a questionnaire.

The system should gradually construct a richer representation of the student.

23. Future Timeline

Once sufficient information is available, the Timeline becomes a major feature.

The student can say naturally:

"Study at 10 AM."

The system converts this into structured task/reminder information.

At 10 AM:

Buddy:
"Are you studying?"

If the student responds:

"No, I'm going to watch a series."

Buddy can reason using historical behaviour:

"Based on your previous pattern, there's a high probability you'll continue watching for another two hours."

The system can then offer:

"Do you want me to show you how this could affect your short-term and long-term goals?"

If the student agrees, the system can generate an alternate timeline.

24. Future Memory Tree

The Memory Tree represents:

Memory
   ↓
Emotion
   ↓
Decision
   ↓
Consequence

For example:

                 Placement Goal
                       │
                       ▼
                Study Decision
                       │
                       ▼
              Chose Entertainment
                       │
                       ▼
              Missed Study Session
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        Frustration             Guilt

The student can eventually use the Reflect feature.

They can select an emotion such as:

Happy
Angry
Sad
Anxious
Frustrated

and Manora can show relevant historical decisions and events associated with that emotional state.

25. Multi-Agent Philosophy

The system is deliberately divided into specialized components.

Emotion Agent

Understands emotions.

Data Agent

Extracts meaningful information.

Memory Engine

Stores and retrieves information.

Buddy State Engine

Controls Buddy's internal state.

Buddy Agent

Decides how Buddy communicates.

Interaction Service

Orchestrates the entire pipeline.

This is preferable to building:

One giant "Manora Agent"

because each component has a clear responsibility and can be tested independently.

26. Technology Responsibilities
FastAPI

Backend API and orchestration.

PostgreSQL / Supabase

Primary structured database.

Stores:

users
sessions
interactions
analyses
goals
memories
Buddy states
Buddy state history
Qdrant

Vector database.

Used for:

Semantic memory retrieval.

Neo4j

Graph database.

Used for:

Relationships between emotions, memories, behaviours, decisions, goals and consequences.

ML Emotion Model

Used for:

Initial emotion signal detection.

LLM through OpenRouter

Used for:

Contextual emotional reasoning and Buddy response generation.

Deterministic State Engine

Used for:

Updating Buddy's internal emotional state.

27. The Core Principle of Manora

The most important idea behind the whole project is:

Manora should not only understand what the student says. It should gradually understand what the student's experiences, emotions, decisions and behaviours mean in the context of their goals.

So the system evolves from:

Conversation

into:

Understanding

and eventually:

Reflection

and:

Behaviour-aware support

The ultimate goal is not to create an AI that tells students what to do.

It is to create a digital companion that can help students recognize patterns between their emotions, decisions, behaviours and goals, and reflect on the consequences of those patterns in a personalized way.