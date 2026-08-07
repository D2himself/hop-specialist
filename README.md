# hop-specialist

**One-liner.** I am distilling a 270 million parameter language model into a specialist for the first step of multi-hop question answering, using HotpotQA.

**Define Hop-1.** Given a multi-hop question, the model outputs the first entity or fact it needs to look up. The output is YAML with three fields, thought, action, and target_entity.

**Eval metric.** I score four things on each output.
- Format adherence: does the output parse as valid YAML.
- Target correctness: does target_entity match one of the gold supporting-fact titles from HotpotQA itself. I judge this with an LLM that treats different phrasings of the same entity as equivalent.
- Exact match: does target_entity match the teacher's target_entity word for word.
- Thought adherence: does the thought field have content.

HotpotQA has no first-hop annotation of its own. Its supporting_facts field is an unordered set of title and sentence id pairs, not an ordered reasoning chain, so it can't tell me which entity belongs at hop 1 versus hop 2. That ordering is what the teacher provides. Target correctness grades against the dataset's gold titles directly. Exact match grades against the teacher's specific choice, since the teacher is the one source that picks out a first hop.

**Results.** I fine-tuned Gemma-3-270M on 379 examples labeled by GPT-4o and evaluated it on a held-out set of 95 questions. I trained two versions to test whether a system prompt helps a model this small. One version had a fixed system turn stating the task and the schema. The other had none. Both used the same training data and the same random split, so the system turn was the only difference between them.

The version with the system turn scored higher on every metric.
- Format adherence: 100% against 98.9%.
- Target correctness: 87.4% against 82.1%.
- Exact match: 66.3% against 61.1%.
- Thought adherence: 100% against 98.9%.

I kept the system prompt in the final design.

**Notebooks.**
- `01_Distill_Reasoning_Gemma3_270M.ipynb`. I load the base model and explore the HotpotQA schema here.
- `02_Teacher_Labeling.ipynb`. I prompt GPT-4o to label each question with its Hop-1 decomposition, then grade the labels with an LLM judge.
- `03_Eval_Harness.ipynb`. I score model outputs on the four metrics above.
- `04_Train_Student_Gemma3_270M.ipynb`. I fine-tune Gemma-3-270M with TRL's SFTTrainer on Google Colab.
