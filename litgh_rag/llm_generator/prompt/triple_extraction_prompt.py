TRIPLE_INSTRUCTIONS = {
    "en":{
        "system": "Always output a valid JSON array only, with no explanations.",
        "entity_relation": """Extract all key entity–relation triples from the passage. Exclude pronouns. Output strictly as:\n
        [
            {
                "Head": "{a noun}",
                "Relation": "{a verb}",
                "Tail": "{a noun}",
            }...
        ]""",

        "event_entity":  """List each event and its participant entities:\n
        [
            {
                "Event": "{a simple sentence describing an event}",
                "Entity": ["entity 1", "entity 2", "..."]
            }...
        ] """,
    
        "event_relation":  """Extract temporal or causal relations between events. Types: before, after, at the same time, because, as a result.:\n
        [
            {
                "Head": "{a simple sentence describing the event 1}",
                "Relation": "{temporal or causality relation between the events}",
                "Tail": "{a simple sentence describing the event 2}"
            }...
        ]""",
        "passage_start" : """Here is the passage."""
    }
}

CONCEPT_INSTRUCTIONS = {
    "en": {
        "event": """Give 3–5 abstract words or short phrases(1–2 words each) describing what this EVENT represents.

            EVENT: A man retreats to mountains and forests.
            Your answer: retreat, relaxation, escape, nature, solitude
            EVENT: A cat chased a prey into its shelter
            Your answer: hunting, escape, predation, hidding, stalking
            EVENT: [EVENT]
            Your answer:""",
        "entity":"""Give 3–5 abstract words or short phrases(1–2 words each) describing what this ENTITY represents.

            ENTITY: Soul
            CONTEXT: premiered BFI London Film Festival, became highest-grossing Pixar release
            Your answer: movie, film

            ENTITY: Thinkpad X60
            CONTEXT: Richard Stallman announced he is using Trisquel on a Thinkpad X60
            Your answer: Thinkpad, laptop, machine, device, hardware, computer, brand

            ENTITY: [ENTITY]
            CONTEXT: [CONTEXT]
            Your answer:""",
        "relation":"""Give 3–5 abstract words or short phrases(1–2 words each) describing what this ENTITY represents.
            
            RELATION: participated in
            Your answer: become part of, attend, take part in, engage in, involve in
            RELATION: be included in
            Your answer: join, be a part of, be a member of, be a component of
            RELATION: [RELATION]
            Your answer:"""
    }
}