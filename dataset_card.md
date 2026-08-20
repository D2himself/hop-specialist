---
license: cc-by-sa-4.0
language:
- en
task_categories:
- text-generation
- question-answering
tags:
- hotpotqa
- multi-hop-reasoning
- distillation
- gpt-4o
- synthetic
size_categories:
- n<1K
---

# hop-1-hotpotqa-decomposition

**What this is.** For each question in this dataset, GPT-4o produced a decomposition of the first reasoning step needed to answer a multi-hop HotpotQA question: what to look up first, not the final answer. I used these labels to fine-tune Gemma-3-270M into a Hop-1 specialist, [Omokemi/hop-1-gemma-270m](https://huggingface.co/Omokemi/hop-1-gemma-270m).

**Source.** Questions come from HotpotQA (Yang et al., 2018), released under CC BY-SA 4.0 since it's built from Wikipedia text. The Hop-1 decomposition for each question, the `thought`, `action`, and `target_entity` fields, comes from GPT-4o given only the question, not the answer or supporting facts. An LLM judge checked each label against HotpotQA's own gold supporting-fact titles before it was kept.

**Schema.**
- `id`: the HotpotQA question id.
- `question`: the original question text.
- `type`: `bridge` or `comparison`, from HotpotQA.
- `level`: `easy`, `medium`, or `hard`, from HotpotQA.
- `gold_titles`: the Wikipedia article titles HotpotQA marks as supporting facts, used to grade `target_entity`.
- `thought`: GPT-4o's reasoning for what to look up first.
- `action`: always `"Lookup"`.
- `target_entity`: the entity or description GPT-4o targets for the first hop.
- `judge_reason`: why the LLM judge accepted this label.
- `messages`: the same fields formatted as a chat-template-ready conversation (`system`/`user`/`assistant` turns), ready to train with TRL's `SFTTrainer`.

**How the labels were generated.** GPT-4o received only the question, never the answer or the supporting facts, and answered with the exact prompt below (five-shot, v4). An LLM judge then checked each `target_entity` for referent equivalence against HotpotQA's own gold supporting-fact titles, accepting a faithful description of an unnamed subject as correct, not only an exact name match. 26 rows were rejected by the judge and are not included here.

```
You are an analytical reasoning agent specialized in breaking down multi-hop questions.

Your objective is to determine ONLY the first logical step (Hop 1) required to solve the question.

CRITICAL CONSTRAINTS:
1. DO NOT answer the question. Stop reasoning immediately after formulating the first hop.
2. Output your response strictly in YAML format. Do not include introductory or concluding text, markdown formatting, or conversational filler.
3. TARGET EXACT ANCHORS: The `target_entity` MUST be taken verbatim from the question. Prefer the explicitly named entity that anchors the lookup. Only when the question names no entity — when it refers to its subject by description alone — use that description verbatim. Never canonicalize, resolve, guess, or inject outside knowledge: do not expand a partial name to its full form, and do not replace a description with the real-world entity it denotes.
4. SEPARATE THE UNKNOWN: Anything you are trying to find out — whether a missing entity or a property — belongs entirely in the `thought`. The `target_entity` is only the known starting bridge the question hands you.
5. STRICT QUOTING: Every value in your YAML output (`thought`, `action`, and `target_entity`) MUST be wrapped in double quotes.

YAML SCHEMA:
thought: "[Your logical deduction of what needs to be found first about the anchor]"
action: "Lookup"
target_entity: "[The exact anchor string handed to you by the question]"

EXAMPLE 1 (Named Entity):
Question: The Oberoi family is part of a hotel company that has a head office in what city?
thought: "I need to find which hotel company the Oberoi family belongs to."
action: "Lookup"
target_entity: "Oberoi family"

EXAMPLE 2 (Comparison):
Question: Were Scott Derrickson and Ed Wood of the same nationality?
thought: "I need to find the nationality of Scott Derrickson first to eventually compare it to Ed Wood."
action: "Lookup"
target_entity: "Scott Derrickson"

EXAMPLE 3 (Anchor Move / Relational):
Question: The wife of Arthur Miller starred in what movie?
thought: "I need to find out who the wife of Arthur Miller is first."
action: "Lookup"
target_entity: "Arthur Miller"

EXAMPLE 4 (Property):
Question: Cadmium Chloride is slightly soluble in this chemical, it is also called what?
thought: "I need to find the chemical that Cadmium Chloride is slightly soluble in."
action: "Lookup"
target_entity: "Cadmium Chloride"

EXAMPLE 5 (Pure Descriptive Anchor):
Question: What language is most widely spoken in the most populous country in Africa?
thought: "I need to identify which country is the most populous in Africa first."
action: "Lookup"
target_entity: "the most populous country in Africa"
```

**Used in.** The `test` split is the benchmark set behind [Omokemi/hop-1-gemma-270m](https://huggingface.co/Omokemi/hop-1-gemma-270m)'s reported results, including a four-way comparison against the untrained base model, Llama-3.1-8B-Instruct, and GPT-4o. Full tables and caveats are on that model's card.

**Splits.** 379 rows in `train.jsonl`, 95 in `test.jsonl`, an 80/20 split with a fixed shuffle seed. Question type holds at roughly 79% bridge and 21% comparison in both splits, matching the full 474-row set (375 bridge, 99 comparison).

**License.** CC BY-SA 4.0, matching HotpotQA's own license, since these questions are HotpotQA's. The `thought`, `action`, and `target_entity` labels are GPT-4o outputs generated for this project.

**Code.** The labeling and grading pipeline that built this dataset is in the [GitHub repo](https://github.com/D2himself/hop-specialist), notebook `02_Teacher_Labeling.ipynb`.
