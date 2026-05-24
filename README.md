# hop-specialist
**One-liner**: Distilling a 270M SLM specialist for first-hop decomposition in multi-hop QA using HotpotQA.

**Define Hop-1**: Given a multi-hop question, output the first logical entity/fact needed, in YAML: {thought: ..., action: ..., target_entity: ...}.

**Eval metric**: Schema adherence (valid YAML %) + Hop-1 accuracy (`target_entity` exact-match against the GPT-4o teacher trace on a held-out HotpotQA split). HotpotQA itself has no "first-hop" annotation — `supporting_facts` is an unordered set of `(title, sent_id)` pairs, not an ordered chain. The gold Hop-1 is teacher-generated.
