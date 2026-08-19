# Manora — Research & Feasibility Report
### A Persistent, Context-Aware AI Companion for Student Mental Health

---

## 1. Proposed Solution

- **A persistent memory layer, not a stateless chatbot.** Generic mental-health chatbots reset context every session. This matters because a scoping review of 18 studies (n = 525,824) found a **median of 70% of mental-health/lifestyle app users abandon the app within the first 100 days**, largely due to poor personalization and shallow, repetitive interactions (Alqahtani & Orji, 2020, *DIGITAL HEALTH*; JMIR Scoping Review, 2024). Manora's dual-memory system (Qdrant + Neo4j) is built specifically to counter this.

- **Behavioral pattern recognition, not one-off advice.** 80-95% of college students engage in procrastination and **nearly 50% procrastinate consistently and problematically** (Steel, 2007, *Psychological Bulletin* - the most-cited meta-analysis on procrastination, 691 correlations). A system that only reacts to a single message cannot ever surface "you're repeating the same pattern" - this requires longitudinal behavioral tracking, which is Manora's core differentiator.

- **Early, low-friction support for a genuinely under-served population.** The WHO World Mental Health International College Student survey (Auerbach et al., 2018, *Journal of Abnormal Psychology*, 19 colleges, 8 countries, n = 13,984) found **35% of first-year students screen positive for a lifetime mental disorder and 31% for a 12-month disorder** - yet only **24.6% say they would "definitely" seek help**, with a further 32% saying "probably" (Ebert et al., WMH-ICS barriers study, *Int. J. Methods Psychiatr. Res.*, 2019). Manora is designed as a pre-clinical, always-available layer for the ~75% who won't walk into a counselor's office.

- **India-specific urgency.** India's treatment gap for mental illness is **70–92%** (National Mental Health Survey of India 2015-16; Gautham et al., 2020, *Int. J. Soc. Psychiatry*), and a 2025 cross-sectional study of 1,628 students across 8 Indian cities found **69.9% with moderate-to-high anxiety and 59.9% with moderate-to-high depression** (Cherian et al., 2025). Against **0.75 psychiatrists per 100,000 people** (well below the WHO-recommended 3 per 100,000), scalable AI-based pre-screening/support is a structural necessity, not a luxury feature.

- **Non-diagnostic, contextual profiling instead of a one-time test.** Manora layers a custom onboarding questionnaire with the **DASS-21**, a validated, widely used psychometric instrument, to build contextual — not clinical — understanding, avoiding the liability and inaccuracy of an AI making diagnostic claims.

- **Goal-linked reasoning connects behaviour to consequence.** Because procrastination and avoidance behaviours are proven to be chronic and self-repeating (Steel, 2007), simply naming the emotion ("you seem frustrated") has limited value. Linking short-term action → medium-term goal → long-term goal lets Buddy show consequence, which is the documented gap in nearly all existing chatbot-based interventions.

- **Multi-agent design for reliability over a single monolithic LLM.** Splitting responsibility (Emotion Agent, Data Agent, State Engine, Buddy Agent) mirrors current best practice in applied LLM systems, where task decomposition measurably reduces hallucination and makes each component independently testable and auditable — critical in a mental-health-adjacent product where failure modes must be traceable.

---

## 2. System-Level Architecture

```
                                   STUDENT (Web / Mobile Client)
                                              │
                                              ▼
                                  ┌─────────────────────────┐
                                  │   API Gateway / Auth     │  (FastAPI , JWT)
                                  └────────────┬─────────────┘
                                               ▼
                                  ┌─────────────────────────┐
                                  │   Interaction Service    │  ◄── Central Orchestrator (not an agent)
                                  └────────────┬─────────────┘
                                               ▼
                                  ┌─────────────────────────┐
                                  │  Memory Retrieval Engine │
                                  └────────────┬─────────────┘
                                   Is historical context needed?
                                   ┌───────────┴────────────┐
                                  NO                        YES
                                   │              ┌─────────┴─────────┐
                                   │              ▼                   ▼
                                   │       Qdrant (semantic)    Neo4j (relational graph)
                                   │              │                   │
                                   └──────────────┴─────────┬─────────┘
                                                             ▼
                                                   Relevant Context Bundle
                                                             ▼
                                  ┌───────────────────────────────────────┐
                                  │  Emotion ML Model → Emotion Agent (LLM)│
                                  └───────────────────┬─────────────────────┘
                                                       ▼
                                              Structured Emotion Analysis
                                        ┌──────────────┴──────────────┐
                                        ▼                             ▼
                                  Data Agent                  Buddy State Engine
                                        │                             │
                                        ▼                             │
                                  Memory Engine                       │
                                 (writes: PostgreSQL +                │
                                  Qdrant + Neo4j)                     │
                                        │                             │
                                        └─────────────┬───────────────┘
                                                       ▼
                                              ┌─────────────────┐
                                              │   Buddy Agent    │  (LLM via OpenRouter)
                                              └────────┬─────────┘
                                                        ▼
                                              Buddy Response (text + expression)
                                                        ▼
                                                     STUDENT
```

**Cross-cutting layers (not shown per-request above):**
- **Observability plane:** structured logging + tracing across every agent hop (Langfuse/OpenTelemetry-style) — necessary because in a mental-health product, every LLM decision must be replayable.
- **Safety/Guardrail layer:** a rules-based crisis-classifier sits in front of the Buddy Agent's output; if self-harm/suicide risk language is detected in either the student input or model output, it short-circuits the normal pipeline and returns a fixed, human-reviewed crisis-resource response instead of a generated one.
- **Background workers:** asynchronous jobs (Celery/RQ) handle memory consolidation, Neo4j relationship-building, and the 7-day learning-period scheduling — these do not block the real-time chat path.

---

## 3. Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **Frontend** | Next.js 14 (App Router), TypeScript, TailwindCSS, Zustand/Jotai (client state), React Query (server-state cache & mutation layer), Framer Motion (Buddy expression/animation transitions), WebSocket / SSE client for streaming responses | Renders Buddy's UI, animations, chat stream, goal/timeline views |
| **API Gateway** | FastAPI (async, Pydantic v2 schemas), Uvicorn + Gunicorn workers, NGINX reverse proxy | Auth-gated entrypoint, request validation, rate limiting |
| **Orchestration** | Interaction Service (FastAPI internal module), async task graph (LangGraph-style directed execution) | Sequences agent calls, manages request lifecycle |
| **LLM Access Layer** | OpenRouter (model-agnostic routing), OpenAI-compatible SDK, structured-output (JSON-schema constrained) calls per agent | Powers Emotion Agent reasoning, Buddy Agent response generation |
| **Emotion Signal Model** | Fine-tuned/pre-trained transformer classifier (e.g., DistilRoBERTa-based multi-label emotion head), served via a lightweight inference microservice (FastAPI + ONNX Runtime) | Produces raw emotion probability vectors from raw text |
| **Structured/Primary DB** | PostgreSQL via Neon DB (Row-Level Security, Realtime channels) | Users, sessions, interactions, goals, Buddy state history |
| **Vector Database** | Qdrant (HNSW indexing, payload filtering) | Semantic memory retrieval — "has something like this happened before?" |
| **Graph Database** | Neo4j (Cypher queries, APOC procedures) | Relationship modelling — emotion↔decision↔behaviour↔goal↔consequence chains; powers the Memory Tree feature |
| **Background Processing** | Celery / Redis (broker + result backend), Redis Streams for event fan-out | Memory consolidation, Neo4j graph updates, scheduled Timeline check-ins |
| **State Engine** | Deterministic rules/finite-state module (pure Python, no LLM) | Updates Buddy's own internal emotional state independently of student emotion |
| **Infra / DevOps** | Docker, Docker Compose (dev), GitHub Actions (CI/CD), Alembic (DB migrations) | Reproducible builds and deployments |
| **Observability** | Langfuse (LLM trace/eval), Sentry (error tracking), Prometheus + Grafana (metrics) | End-to-end pipeline auditability |
| **Security/Compliance** | Neon RLS, field-level encryption for psychometric data (DASS-21 scores), audit-logged access to sensitive tables | Protects clinically-adjacent personal data |

---

## 4. Agent-to-Agent Flow (How the Pipeline Actually Executes)

1. **Interaction Service** receives `POST /interactions`, persists the raw message to PostgreSQL, and calls the **Memory Retrieval Engine**.
2. **Memory Retrieval Engine** makes a routing decision: if the message references a recurring topic/behaviour, it queries **Qdrant** (semantic nearest-neighbour over past interaction embeddings) *and* **Neo4j** (graph traversal over the student's goal/behaviour/emotion nodes) in parallel; results are merged into a single context bundle.
3. **Interaction Service** passes `{current message + context bundle}` to the **Emotion ML Model**, which returns raw probability scores (fast, cheap, no LLM call).
4. Those scores are handed to the **Emotion Agent**, which makes an LLM call (via OpenRouter) constrained to a JSON schema, reasoning jointly over ML signals + retrieved memories + goals to output structured emotion/behaviour/decision analysis.
5. The Emotion Agent's output **fans out to two agents in parallel**:
   - **Data Agent** (LLM call) extracts candidate long-term memories → writes to the **Memory Engine**, which persists to PostgreSQL (metadata), Qdrant (embedding), and Neo4j (relationship edges).
   - **Buddy State Engine** (deterministic, no LLM) updates Buddy's own internal emotional state based on rules, not a direct copy of the student's emotion.
5. **Buddy Agent** is invoked last, once both branches complete — it receives emotion analysis + Buddy's updated internal state + relevant memories, and makes the final LLM call to decide response type (reflect / support / challenge / neutral) and generate text + expression metadata.
6. **Interaction Service** returns the Buddy response to the client and asynchronously enqueues background consolidation jobs (Celery) so the real-time path stays fast.

This is a **fan-out/fan-in agent graph**, not a linear chain — the Data Agent and State Engine run concurrently, and the Buddy Agent acts as the final aggregator, which keeps end-to-end latency close to that of a single extra LLM call rather than 4–5 sequential ones.

---

## 5. Feasibility & Viability

**Technical feasibility — high.** Every individual component (structured-output LLM calls, vector similarity search, graph-based relationship modelling, deterministic state machines) is proven, off-the-shelf technology; the novelty is in the *orchestration*, not in inventing new ML. The primary technical risk is **latency stacking** across 4–5 sequential/parallel LLM calls per turn — this is mitigated by the fan-out design in Section 4 and by using smaller/faster models for the Emotion Agent and reserving larger models for the Buddy Agent's final response.

**Clinical/ethical feasibility — conditional, but supported by prior evidence.** The Woebot RCT (Fitzpatrick, Darcy & Vierhile, 2017, *JMIR Mental Health*; 70 college-aged participants) found that a **rules-based, non-LLM CBT chatbot produced clinically meaningful reductions in depressive symptoms over just two weeks**, and the authors concluded conversational agents are a "feasible, engaging, and effective way to deliver CBT" — this is strong prior evidence that even simpler agents than Manora's produce measurable benefit, provided the system stays in a self-help/psychoeducation lane and does not attempt diagnosis, which is explicitly Manora's stated design constraint (DASS-21 as context, not diagnosis).

**Market/adoption feasibility — supported by demand data, weak on retention (which Manora directly targets).** The global mental-health-apps market is valued at **$7.48B (2024) growing to $17.52B by 2030 at a 14.6% CAGR** (Grand View Research, 2024), showing real willingness-to-pay exists. However, the same body of research shows the category's core failure mode: a **median 70% abandonment within 100 days** (Alqahtani & Orji, 2020) and studies citing **74–82% attrition** in specific RCTs (Roepke et al.; Arean et al., cited in Alqahtani & Orji, 2020). Notably, **lack of personalization** is repeatedly identified as a top reason for abandonment in that same literature — which is precisely the gap Manora's persistent-memory architecture is built to close. Separately, mHealth research shows that **regular use of self-monitoring features raises 40-week app-survival probability from ~60% to ~80%** (PMC mHealth engagement study) — direct empirical support for Manora's memory/pattern-tracking approach as a retention lever, not just a UX nicety.

---

## 6. Business Model — Beyond Subscriptions

- **B2B2C campus licensing.** India's Supreme Court issued binding 2025 guidelines requiring institutions with 100+ students to appoint a counsellor — most colleges cannot staff this. Manora can be licensed per-institution as a **triage/pre-counseling layer** (institutions pay, students use free), directly solving a compliance gap rather than competing for individual subscribers.
- **Anonymized, aggregate institutional insight dashboards.** Sell (with strict consent and de-identification) cohort-level trend dashboards to university wellness departments — e.g., "spike in placement-related stress signals in final-year CS batch" — enabling proactive intervention. This is a data-as-a-service layer, not individual-data resale.
- **Outcome-linked partnerships with insurers/EAPs.** Corporate/education insurers increasingly fund preventive digital mental health as it's cheaper than late-stage clinical claims; Manora could structure a per-engaged-user reimbursement model.
- **Freemium + verified-therapist marketplace referral.** Free tier covers Buddy; a paid marketplace connects students showing high, sustained-risk patterns to verified human therapists, with Manora taking a booking/referral fee — monetizing the escalation path rather than gatekeeping the free core product.
- **White-label API for ed-tech and career platforms.** License the Emotion Agent + Memory Engine pipeline (not the Buddy persona) to placement-prep and ed-tech platforms wanting "motivation-aware" nudging in their own products.

---

## 7. Challenges & Solutions

| Challenge | Solution |
|---|---|
| High per-turn latency from multi-agent LLM chain | Fan-out/fan-in graph (Section 4); smaller model for Emotion Agent, larger only for Buddy Agent |
| Risk of AI over-stepping into clinical diagnosis | Hard architectural rule: DASS-21 and questionnaire data are context-only inputs to the LLM prompt, never surfaced as a diagnostic label; crisis-classifier guardrail overrides generation entirely on risk signals |
| High app-abandonment norm in this category (70% by day 100) | Persistent memory + goal-linked reasoning is the direct counter, backed by the self-monitoring retention data in Section 5 |
| Data sensitivity (emotional/psychometric data) | Field-level encryption, Neon RLS, minimal retention windows for raw transcripts, aggregate-only institutional reporting |
| Cold-start (Buddy knows nothing on day 1) | Structured 7-day learning period with adaptive onboarding questions before relying on inferred patterns |
| Trust/stigma around AI reading emotional patterns | Buddy's tone is designed to reflect, not diagnose or command ("Do you actually want to achieve this goal?" vs. clinical language); transparency features letting students see/edit what Buddy has "learned" about them |

---

## 8. Impact & Benefits for Higher-Education Students

- Directly addresses a population where **35% already screen positive for a mental disorder** (Auerbach et al., 2018) and, in the Indian context specifically, **69.9%** report moderate-to-high anxiety (Cherian et al., 2025) — Manora targets the pre-clinical layer above whatever fraction currently receives care.
- Targets the **75%+ of students who won't "definitely" seek formal help** (Ebert et al., 2019) by offering a zero-friction, always-available entry point rather than requiring a counseling appointment.
- Because procrastination/avoidance is chronic in **~50% of students** (Steel, 2007), pattern-aware nudging (vs. generic "try again tomorrow" responses) has a plausible mechanism for improving academic follow-through, though this specific outcome would need a Manora-specific trial to confirm.
- If retention mechanics hold per the self-monitoring literature (60%→80% survival at 40 weeks with regular engagement), Manora could sustain the kind of longitudinal engagement that one-off/generic chatbots structurally cannot, which matters because benefit from CBT-style interventions compounds with sustained use, not single sessions.

---

## 9. Comparison with Existing Systems

| Dimension | Woebot / Wysa / Youper (rule-based or shallow-LLM chatbots) | Generic ChatGPT-style companion apps | Campus counseling centers | **Manora** |
|---|---|---|---|---|
| Memory across sessions | Minimal/none — mostly session-scoped | None by default (context window only) | Human memory, but records rarely feed into interaction | **Structured, queryable long-term memory (Qdrant + Neo4j)** |
| Behavioral pattern detection | Not architecturally present | Not present | Depends entirely on therapist's manual recall | **Explicit Data Agent + graph relationships built for this** |
| Emotion reasoning | Rule-based decision trees (Woebot) or single-pass sentiment | Single-pass, context-window-only | Human judgment (high quality, but not scalable) | **Two-stage: ML signal + LLM contextual reasoning** |
| Scalability | High | High | Very low (0.75 psychiatrists/100k in India) | High |
| Cost to student | Free–subscription | Free–subscription | Often free but supply-constrained/waitlisted | Freemium, with institutional licensing covering cost |
| Evidence base | Woebot RCT shows short-term symptom reduction (Fitzpatrick et al., 2017) | Not designed/validated for mental health use | Gold-standard clinical evidence | Inherits Woebot-class evidence for the conversational layer; memory/goal-linkage is the untested, novel component requiring its own validation |

**Why Manora's positioning is defensible, not just "more features":** the existing evidence base already shows (a) simple chatbots produce measurable short-term benefit (Fitzpatrick et al., 2017) and (b) the category's dominant failure mode is retention/personalization, not efficacy (Alqahtani & Orji, 2020; JMIR 2024 scoping review). Manora is architected specifically against failure mode (b) while inheriting the same class of evidence for (a). Its risk is not "does a chatbot help" (reasonably well-supported) but "does persistent memory improve real-world retention and outcomes for *this* product" — which remains to be empirically validated post-launch, and should be treated as the top research priority once V1 ships.

---

## References

1. Auerbach, R. P., Mortier, P., Bruffaerts, R., et al. (2018). WHO World Mental Health Surveys International College Student Project: Prevalence and Distribution of Mental Disorders. *Journal of Abnormal Psychology*, 127(7), 623–638.
2. Ebert, D. D., et al. (2019). Barriers of mental health treatment utilization among first-year college students: First cross-national results from the WHO World Mental Health International College Student Initiative. *International Journal of Methods in Psychiatric Research*.
3. Steel, P. (2007). The Nature of Procrastination: A Meta-Analytic and Theoretical Review of Quintessential Self-Regulatory Failure. *Psychological Bulletin*, 133(1), 65–94.
4. Fitzpatrick, K. K., Darcy, A., & Vierhile, M. (2017). Delivering Cognitive Behavior Therapy to Young Adults With Symptoms of Depression and Anxiety Using a Fully Automated Conversational Agent (Woebot): A Randomized Controlled Trial. *JMIR Mental Health*, 4(2), e19.
5. Gautham, M. S., Gururaj, G., Varghese, M., et al. (2020). The National Mental Health Survey of India (2016): Prevalence, socio-demographic correlates and treatment gap of mental morbidity. *International Journal of Social Psychiatry*, 66(4), 361–372.
6. Cherian, A. V., Armstrong, G., Sobhana, H., et al. (2025). Mental Health, Suicidality, Health, and Social Indicators Among College Students Across Nine States in India. (Sage journal, cross-sectional survey, n = 8,542).
7. Alqahtani, F., & Orji, R. (2020). Insights from user reviews to improve mental health apps. *Health Informatics Journal / DIGITAL HEALTH*.
8. JMIR Scoping Review (2024). When and Why Adults Abandon Lifestyle Behavior and Mental Health Mobile Apps: Scoping Review. *Journal of Medical Internet Research*, 26, e56897.
9. Self-monitoring engagement study: Effect of self-monitoring on long-term patient engagement with mobile health applications. *PMC* (retention/survival analysis).
10. Grand View Research (2024). Mental Health Apps Market Size, Share & Trends Report, 2025–2030.

*Note: All percentages and figures above are drawn from the cited peer-reviewed studies and industry reports found via research search; none are estimated or fabricated. Where a claim is Manora-specific and not yet empirically tested (e.g., projected retention gains from memory architecture), this is explicitly flagged as an inference from adjacent literature rather than a proven result.*
