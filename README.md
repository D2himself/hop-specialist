# hop-specialist
**One-liner**: Distilling a 270M SLM specialist for first-hop decomposition in multi-hop QA using HotpotQA.

**Define Hop-1**: Given a multi-hop question, output the first logical entity/fact needed, in YAML: {thought: ..., action: ..., target_entity: ...}.

**Eval metric**: Schema adherence (valid YAML %) + Hop-1 accuracy (exact match on target_entity vs. gold first-hop from HotpotQA annotations).
