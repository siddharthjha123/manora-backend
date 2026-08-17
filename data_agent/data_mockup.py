"""
Manora - Data Agent Mock Dataset

Purpose:
    Development dataset for testing the Data Agent's
    long-term memory processing.

Student profile:
    A university student who is socially isolated and
    frequently experiences loneliness.

Dataset:
    20 interactions
    16 candidate memories
    4 interactions intentionally have no candidate memory

Important:
    These are MOCK observations for development/testing.
    They are not psychological diagnoses.
"""

MOCK_USER = {
    "user_id": "user_001",
    "session_id": "session_001",
}


MOCK_INTERACTIONS = [
    # =========================================================
    # INTERACTION 1
    # Candidate memory: YES
    # =========================================================

    {
        "interaction_id": "int_001",
        "user_id": "user_001",
        "session_id": "session_001",
        "role": "user",
        "raw_text": (
            "Honestly bro, college feels really lonely. "
            "I go to classes, come back to my room and that's basically my whole day."
        ),
        "candidate_memory": {
            "memory_id": "cm_001",
            "content": (
                "Student experiences college as a lonely environment "
                "and spends most of their time between classes and their room."
            ),
            "context": {
                "domain": "social",
                "subtopic": "loneliness"
            },
            "emotional_state": [
                {
                    "emotion": "loneliness",
                    "confidence": 0.91,
                    "source": "user_explicit_context"
                },
                {
                    "emotion": "sadness",
                    "confidence": 0.63,
                    "source": "model_inferred"
                }
            ],
            "importance": 0.86,
            "confidence": 0.91
        }
    },


    # =========================================================
    # INTERACTION 2
    # Candidate memory: YES
    # =========================================================

    {
        "interaction_id": "int_002",
        "user_id": "user_001",
        "session_id": "session_001",
        "role": "user",
        "raw_text": (
            "There are people around me obviously, but I don't really "
            "have anyone I can just call and talk to when something happens."
        ),
        "candidate_memory": {
            "memory_id": "cm_002",
            "content": (
                "Student has people around them but lacks a close person "
                "they feel comfortable calling or talking to."
            ),
            "context": {
                "domain": "social",
                "subtopic": "social_support"
            },
            "emotional_state": [
                {
                    "emotion": "loneliness",
                    "confidence": 0.89,
                    "source": "model_inferred"
                }
            ],
            "importance": 0.88,
            "confidence": 0.92
        }
    },


    # =========================================================
    # INTERACTION 3
    # Candidate memory: NO
    # =========================================================

    {
        "interaction_id": "int_003",
        "user_id": "user_001",
        "session_id": "session_001",
        "role": "user",
        "raw_text": (
            "Anyway, I'm going to grab some food now. "
            "I'll probably order something."
        ),
        "candidate_memory": None
    },


    # =========================================================
    # INTERACTION 4
    # Candidate memory: YES
    # =========================================================

    {
        "interaction_id": "int_004",
        "user_id": "user_001",
        "session_id": "session_001",
        "role": "user",
        "raw_text": (
            "Sometimes I open Instagram just to see what everyone else "
            "is doing. Then I see people hanging out together and I feel "
            "like I'm missing out on everything."
        ),
        "candidate_memory": {
            "memory_id": "cm_004",
            "content": (
                "Student uses Instagram to check what others are doing "
                "and experiences social comparison and fear of missing out "
                "when seeing others socializing."
            ),
            "context": {
                "domain": "digital_life",
                "subtopic": "social_comparison_fomo"
            },
            "emotional_state": [
                {
                    "emotion": "loneliness",
                    "confidence": 0.79,
                    "source": "model_inferred"
                },
                {
                    "emotion": "fomo",
                    "confidence": 0.88,
                    "source": "user_explicit_context"
                }
            ],
            "importance": 0.84,
            "confidence": 0.90
        }
    },


    # =========================================================
    # INTERACTION 5
    # Candidate memory: YES
    # =========================================================

    {
        "interaction_id": "int_005",
        "user_id": "user_001",
        "session_id": "session_001",
        "role": "user",
        "raw_text": (
            "And then I start thinking that everyone is moving ahead "
            "while I'm just sitting in my room doing nothing."
        ),
        "candidate_memory": {
            "memory_id": "cm_005",
            "content": (
                "Student compares their progress with others and feels "
                "that they are falling behind while spending time alone."
            ),
            "context": {
                "domain": "personal",
                "subtopic": "social_comparison"
            },
            "emotional_state": [
                {
                    "emotion": "insecurity",
                    "confidence": 0.78,
                    "source": "model_inferred"
                },
                {
                    "emotion": "sadness",
                    "confidence": 0.73,
                    "source": "model_inferred"
                }
            ],
            "importance": 0.81,
            "confidence": 0.84
        }
    },


    # =========================================================
    # INTERACTION 6
    # Candidate memory: YES
    # =========================================================

    {
        "interaction_id": "int_006",
        "user_id": "user_001",
        "session_id": "session_001",
        "role": "user",
        "raw_text": (
            "My grades aren't even that bad, but I can't concentrate "
            "properly when I'm studying alone. I end up checking my phone."
        ),
        "candidate_memory": {
            "memory_id": "cm_006",
            "content": (
                "Student finds it difficult to maintain concentration "
                "while studying alone and frequently checks their phone."
            ),
            "context": {
                "domain": "education",
                "subtopic": "study_concentration"
            },
            "emotional_state": [
                {
                    "emotion": "frustration",
                    "confidence": 0.68,
                    "source": "model_inferred"
                }
            ],
            "importance": 0.75,
            "confidence": 0.86
        }
    },


    # =========================================================
    # INTERACTION 7
    # Candidate memory: NO
    # =========================================================

    {
        "interaction_id": "int_007",
        "user_id": "user_001",
        "session_id": "session_001",
        "role": "user",
        "raw_text": (
            "Oh and I think I have a lecture at 11 tomorrow."
        ),
        "candidate_memory": None
    },


    # =========================================================
    # INTERACTION 8
    # Candidate memory: YES
    # =========================================================

    {
        "interaction_id": "int_008",
        "user_id": "user_001",
        "session_id": "session_001",
        "role": "user",
        "raw_text": (
            "I haven't really talked to my family about any of this. "
            "I don't want them to worry about me."
        ),
        "candidate_memory": {
            "memory_id": "cm_008",
            "content": (
                "Student avoids discussing their difficulties with family "
                "because they do not want their family to worry."
            ),
            "context": {
                "domain": "family",
                "subtopic": "emotional_support"
            },
            "emotional_state": [
                {
                    "emotion": "worry",
                    "confidence": 0.65,
                    "source": "model_inferred"
                },
                {
                    "emotion": "isolation",
                    "confidence": 0.71,
                    "source": "model_inferred"
                }
            ],
            "importance": 0.79,
            "confidence": 0.86
        }
    },


    # =========================================================
    # INTERACTION 9
    # Candidate memory: YES
    # =========================================================

    {
        "interaction_id": "int_009",
        "user_id": "user_001",
        "session_id": "session_001",
        "role": "user",
        "raw_text": (
            "I actually want to get a good placement after graduation. "
            "That's probably the one thing I'm seriously worried about."
        ),
        "candidate_memory": {
            "memory_id": "cm_009",
            "content": (
                "Student has a goal of securing a good placement after "
                "graduation and is seriously concerned about achieving it."
            ),
            "context": {
                "domain": "career",
                "subtopic": "placement"
            },
            "emotional_state": [
                {
                    "emotion": "stress",
                    "confidence": 0.83,
                    "source": "user_explicit_context"
                }
            ],
            "goal": {
                "description": "Secure a good placement after graduation",
                "type": "career"
            },
            "importance": 0.94,
            "confidence": 0.94
        }
    },


    # =========================================================
    # INTERACTION 10
    # Candidate memory: YES
    # =========================================================

    {
        "interaction_id": "int_010",
        "user_id": "user_001",
        "session_id": "session_001",
        "role": "user",
        "raw_text": (
            "The placement thing gets worse when I see people on LinkedIn "
            "posting internships and offers. I start wondering if I'm already behind."
        ),
        "candidate_memory": {
            "memory_id": "cm_010",
            "content": (
                "Seeing other students' internships and job offers on "
                "LinkedIn increases the student's concern about falling behind "
                "in their career."
            ),
            "context": {
                "domain": "career",
                "subtopic": "career_comparison"
            },
            "emotional_state": [
                {
                    "emotion": "stress",
                    "confidence": 0.82,
                    "source": "model_inferred"
                },
                {
                    "emotion": "insecurity",
                    "confidence": 0.77,
                    "source": "model_inferred"
                }
            ],
            "importance": 0.87,
            "confidence": 0.89
        }
    },


    # =========================================================
    # INTERACTION 11
    # Candidate memory: YES
    # =========================================================

    {
        "interaction_id": "int_011",
        "user_id": "user_001",
        "session_id": "session_001",
        "role": "user",
        "raw_text": (
            "Sometimes I stay awake until like 2 or 3 because my brain "
            "just keeps replaying everything. College, career, friends, all of it."
        ),
        "candidate_memory": {
            "memory_id": "cm_011",
            "content": (
                "Student frequently stays awake late because they repeatedly "
                "think about college, career and social concerns."
            ),
            "context": {
                "domain": "personal",
                "subtopic": "sleep_and_overthinking"
            },
            "emotional_state": [
                {
                    "emotion": "stress",
                    "confidence": 0.81,
                    "source": "model_inferred"
                },
                {
                    "emotion": "anxiety",
                    "confidence": 0.75,
                    "source": "model_inferred"
                }
            ],
            "importance": 0.88,
            "confidence": 0.86
        }
    },


    # =========================================================
    # INTERACTION 12
    # Candidate memory: NO
    # =========================================================

    {
        "interaction_id": "int_012",
        "user_id": "user_001",
        "session_id": "session_001",
        "role": "user",
        "raw_text": (
            "Bro that food I ordered yesterday was actually terrible 😂."
        ),
        "candidate_memory": None
    },


    # =========================================================
    # INTERACTION 13
    # Candidate memory: YES
    # =========================================================

    {
        "interaction_id": "int_013",
        "user_id": "user_001",
        "session_id": "session_001",
        "role": "user",
        "raw_text": (
            "I used to have one really close friend in school. "
            "We talked every day, but after college we just drifted apart."
        ),
        "candidate_memory": {
            "memory_id": "cm_013",
            "content": (
                "Student previously had a very close friendship in school "
                "that gradually became distant after entering college."
            ),
            "context": {
                "domain": "past_life",
                "subtopic": "friendship_loss"
            },
            "emotional_state": [
                {
                    "emotion": "sadness",
                    "confidence": 0.78,
                    "source": "model_inferred"
                },
                {
                    "emotion": "nostalgia",
                    "confidence": 0.65,
                    "source": "model_inferred"
                }
            ],
            "importance": 0.82,
            "confidence": 0.88
        }
    },


    # =========================================================
    # INTERACTION 14
    # Candidate memory: YES
    # =========================================================

    {
        "interaction_id": "int_014",
        "user_id": "user_001",
        "session_id": "session_001",
        "role": "user",
        "raw_text": (
            "I think that's why I don't trust people that easily anymore. "
            "I don't want to get close to someone and then lose them again."
        ),
        "candidate_memory": {
            "memory_id": "cm_014",
            "content": (
                "Student is hesitant to form close relationships because "
                "of a previous friendship that became distant."
            ),
            "context": {
                "domain": "social",
                "subtopic": "trust_and_relationships"
            },
            "emotional_state": [
                {
                    "emotion": "fear",
                    "confidence": 0.78,
                    "source": "model_inferred"
                },
                {
                    "emotion": "sadness",
                    "confidence": 0.69,
                    "source": "model_inferred"
                }
            ],
            "importance": 0.87,
            "confidence": 0.84
        }
    },


    # =========================================================
    # INTERACTION 15
    # Candidate memory: YES
    # =========================================================

    {
        "interaction_id": "int_015",
        "user_id": "user_001",
        "session_id": "session_001",
        "role": "user",
        "raw_text": (
            "When I feel like this I usually just put on headphones "
            "and disappear into YouTube for hours. At least then I don't think."
        ),
        "candidate_memory": {
            "memory_id": "cm_015",
            "content": (
                "Student uses prolonged YouTube consumption as a way to "
                "avoid thinking about difficult emotions or problems."
            ),
            "context": {
                "domain": "digital_life",
                "subtopic": "digital_coping"
            },
            "emotional_state": [
                {
                    "emotion": "overwhelm",
                    "confidence": 0.76,
                    "source": "model_inferred"
                },
                {
                    "emotion": "loneliness",
                    "confidence": 0.72,
                    "source": "session_context"
                }
            ],
            "importance": 0.90,
            "confidence": 0.87
        }
    },


    # =========================================================
    # INTERACTION 16
    # Candidate memory: YES
    # =========================================================

    {
        "interaction_id": "int_016",
        "user_id": "user_001",
        "session_id": "session_001",
        "role": "user",
        "raw_text": (
            "But honestly I don't want to stay like this. "
            "I want to become good at ML, get a job I actually like, "
            "and have a few people around me who genuinely care."
        ),
        "candidate_memory": {
            "memory_id": "cm_016",
            "content": (
                "Student wants to improve their machine-learning skills, "
                "secure meaningful employment and build genuine social connections."
            ),
            "context": {
                "domain": "personal",
                "subtopic": "future_aspirations"
            },
            "goals": [
                {
                    "description": "Become good at machine learning",
                    "type": "personal_education"
                },
                {
                    "description": "Get a meaningful job",
                    "type": "career"
                },
                {
                    "description": "Build genuine social connections",
                    "type": "social"
                }
            ],
            "emotional_state": [
                {
                    "emotion": "hope",
                    "confidence": 0.78,
                    "source": "model_inferred"
                }
            ],
            "importance": 0.96,
            "confidence": 0.92
        }
    },


    # =========================================================
    # INTERACTION 17
    # Candidate memory: NO
    # =========================================================

    {
        "interaction_id": "int_017",
        "user_id": "user_001",
        "session_id": "session_001",
        "role": "user",
        "raw_text": (
            "Anyway, tomorrow I'll probably wake up early. "
            "Let's see how that goes."
        ),
        "candidate_memory": None
    },


    # =========================================================
    # INTERACTION 18
    # Candidate memory: YES
    # =========================================================

    {
        "interaction_id": "int_018",
        "user_id": "user_001",
        "session_id": "session_001",
        "role": "user",
        "raw_text": (
            "Actually when I go to the library and there are other people "
            "studying around me, I focus much better. I don't feel as distracted."
        ),
        "candidate_memory": {
            "memory_id": "cm_018",
            "content": (
                "Student focuses better when studying around other people "
                "in a library compared with studying alone."
            ),
            "context": {
                "domain": "education",
                "subtopic": "study_environment"
            },
            "emotional_state": [
                {
                    "emotion": "calm",
                    "confidence": 0.61,
                    "source": "model_inferred"
                }
            ],
            "behavior": {
                "type": "sustained_focus",
                "context": "social_study_environment"
            },
            "importance": 0.89,
            "confidence": 0.91
        }
    },


    # =========================================================
    # INTERACTION 19
    # Candidate memory: YES
    # =========================================================

    {
        "interaction_id": "int_019",
        "user_id": "user_001",
        "session_id": "session_001",
        "role": "user",
        "raw_text": (
            "Maybe I don't actually hate being around people. "
            "I think I just don't know how to make close friends anymore."
        ),
        "candidate_memory": {
            "memory_id": "cm_019",
            "content": (
                "Student believes they may want social connection but "
                "finds it difficult to form close friendships."
            ),
            "context": {
                "domain": "social",
                "subtopic": "social_connection"
            },
            "emotional_state": [
                {
                    "emotion": "loneliness",
                    "confidence": 0.82,
                    "source": "session_context"
                },
                {
                    "emotion": "uncertainty",
                    "confidence": 0.71,
                    "source": "model_inferred"
                }
            ],
            "importance": 0.93,
            "confidence": 0.89
        }
    },


    # =========================================================
    # INTERACTION 20
    # Candidate memory: YES
    # =========================================================

    {
        "interaction_id": "int_020",
        "user_id": "user_001",
        "session_id": "session_001",
        "role": "user",
        "raw_text": (
            "If I could fix one thing right now, I'd want to stop feeling "
            "so alone all the time. I think everything else would become "
            "a little easier if I had someone I could genuinely talk to."
        ),
        "candidate_memory": {
            "memory_id": "cm_020",
            "content": (
                "Student identifies persistent loneliness and genuine "
                "social connection as an important area they want to improve."
            ),
            "context": {
                "domain": "social",
                "subtopic": "loneliness_and_support"
            },
            "emotional_state": [
                {
                    "emotion": "loneliness",
                    "confidence": 0.95,
                    "source": "user_explicit"
                },
                {
                    "emotion": "hope",
                    "confidence": 0.63,
                    "source": "model_inferred"
                }
            ],
            "goal": {
                "description": "Develop genuine social connection and reduce persistent loneliness",
                "type": "social"
            },
            "importance": 0.98,
            "confidence": 0.96
        }
    }
]


# =============================================================
# Helper functions
# =============================================================

def get_all_interactions():
    """Return all mock interactions."""
    return MOCK_INTERACTIONS


def get_candidate_memories():
    """Return only interactions that contain candidate memories."""
    return [
        interaction["candidate_memory"]
        for interaction in MOCK_INTERACTIONS
        if interaction["candidate_memory"] is not None
    ]


def get_candidate_memory_models():
    """Return the existing mock candidates as the production input contract.

    The mock fixture predates ``CandidateMemory`` and intentionally retains its
    original dictionary shape.  This adapter only translates those existing 16
    records; it does not add or invent any observations.
    """
    from data_agent.data_schema import (
        CandidateMemory,
        CandidateMemoryBehavior,
        CandidateMemoryEmotion,
        CandidateMemoryGoalRelevance,
    )

    candidates = []
    for interaction in MOCK_INTERACTIONS:
        raw = interaction["candidate_memory"]
        if raw is None:
            continue

        raw_behavior = raw.get("behavior")
        behavior = None
        if raw_behavior:
            behavior = CandidateMemoryBehavior(
                type=raw_behavior.get("type", "observed_behavior"),
                description=raw_behavior.get("description")
                or raw_behavior.get("context")
                or raw["content"],
            )

        goal_descriptions = []
        if raw.get("goal"):
            goal_descriptions.append(raw["goal"].get("description", ""))
        for goal in raw.get("goals", []):
            goal_descriptions.append(goal.get("description", ""))
        goal_descriptions = [goal for goal in goal_descriptions if goal]

        candidates.append(
            CandidateMemory(
                id=raw["memory_id"],
                user_id=interaction["user_id"],
                content=raw["content"],
                context=dict(raw.get("context", {})),
                emotional_state=[
                    CandidateMemoryEmotion(
                        emotion=item["emotion"],
                        confidence=item["confidence"],
                    )
                    for item in raw.get("emotional_state", [])
                ],
                behavior=behavior,
                goal_relevance=CandidateMemoryGoalRelevance(
                    related=bool(goal_descriptions),
                    goal="; ".join(goal_descriptions) if goal_descriptions else None,
                ),
                importance=raw.get("importance", 0.5),
                confidence=raw.get("confidence", 0.5),
            )
        )

    return candidates


def get_non_memory_interactions():
    """Return interactions that should not become memories."""
    return [
        interaction
        for interaction in MOCK_INTERACTIONS
        if interaction["candidate_memory"] is None
    ]


def get_interaction_count():
    """Return total number of interactions."""
    return len(MOCK_INTERACTIONS)


def get_candidate_memory_count():
    """Return total number of candidate memories."""
    return len(get_candidate_memories())


if __name__ == "__main__":
    print("Total interactions:", get_interaction_count())
    print("Candidate memories:", get_candidate_memory_count())
    print(
        "Non-memory interactions:",
        len(get_non_memory_interactions())
    )
