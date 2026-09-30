# Archived by llm_client Plan #36

`debate_agent.py` is preserved as generated prior art for the intermodel
collaboration work that now lives in `llm_client`.

Do not extend this generated agent as the active debate or review runtime. The
useful concepts were extracted into `llm_client` Plan #36 prior-art notes:

- moderator setup of a proposition and opposed positions
- pro/con bounded-round debate topology
- turn-level history records
- separate judge/adjudicator synthesis
- schema-compliance reporting
- trace and execution metrics

Canonical replacement surfaces:

- `llm_client deliberate-task` for symmetric debate and model deliberation
- Plan #35 barrier/anonymization for preserving independent turns
- Plan #34 verifier/adjudicator ledgers for synthesis and claim review
- `llm_client review-cycle` for artifact review/apply loops
- `prompt_eval` for frozen-case comparisons and debate-quality metrics

The generated file remains in place so references do not break. New work
should happen in the canonical replacement surfaces above.

