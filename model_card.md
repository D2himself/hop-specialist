---
license: gemma
base_model: google/gemma-3-270m-it
tags:
- gemma
- text-generation
- multi-hop-qa
- distillation
- reasoning
- hotpotqa
datasets:
- Omokemi/hop-1-hotpotqa-decomposition
language:
- en
pipeline_tag: text-generation
---

# hop-1-gemma-270m

**One-liner.** I fine-tuned Gemma-3-270M-it into a specialist for the first step of multi-hop question answering, using HotpotQA. Given a question, it outputs the first thing to look up, not the final answer.

**Define Hop-1.** Given a multi-hop question, the model outputs the first entity or fact it needs to look up. The output is YAML with three fields: `thought`, `action`, and `target_entity`. It does not retrieve documents, run the second hop, or produce a final answer.

**How I built it.** GPT-4o labeled 474 HotpotQA questions with their Hop-1 decomposition. I fine-tuned Gemma-3-270M-it on 379 of those labels with TRL's `SFTTrainer` and held out 95 for evaluation. The full labeling and training pipeline is in the [GitHub repo](https://github.com/D2himself/hop-specialist).

**Prompt format.** This model expects a fixed system turn stating the task and the schema, plus the question as the user turn.

System turn:
```
You are a reasoning agent. Given a multi-hop question, output only the first reasoning step as YAML with three fields: thought, action, target_entity.
```

Usage:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Omokemi/hop-1-gemma-270m")
tokenizer = AutoTokenizer.from_pretrained("Omokemi/hop-1-gemma-270m")
model.eval()  # required, see the note below

messages = [
    {"role": "system", "content": "You are a reasoning agent. Given a multi-hop question, output only the first reasoning step as YAML with three fields: thought, action, target_entity."},
    {"role": "user", "content": "Were Scott Derrickson and Ed Wood of the same nationality?"},
]

inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
output = model.generate(inputs, max_new_tokens=100)
print(tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True))
```

**Call `model.eval()` before generating.** Loaded fresh from `from_pretrained`, the model starts in train mode. Generating with dropout still active on a model this small can collapse into a repeated-character loop instead of a real answer. This is a bug I actually hit during training, not a hypothetical one.

**Results.** I trained two versions to test whether a fixed system prompt helps a model this small: this one, with the system turn above, and a second version with none. Both used the same training data and the same random split, so the system turn is the only difference. Scored on the 95 held-out questions:

| Metric | With system prompt | Without |
|---|---|---|
| Format adherence | 100% | 98.9% |
| Target correctness | 87.4% | 82.1% |
| Exact match to teacher | 66.3% | 61.1% |
| Thought adherence | 100% | 98.9% |

The version with the system prompt won on every metric, so that's the version published here.

A note on the numbers below: the benchmark in the next section used a fresh retrain of this same sys arm, on the same data and setup, run in a separate session. Its target correctness came out at 85.3% and its exact match at 62.1%, a few points under the 87.4% and 66.3% above. Neither run used a fixed training seed, so this is normal run-to-run variation, not a different model.

**How it compares to bigger models.** I benchmarked this model against the same base checkpoint with no fine-tuning, Llama-3.1-8B-Instruct through Groq, and the GPT-4o teacher that produced the training labels, all on the same 95 held-out questions. `untrained` and this model used the short system prompt shown above. `groq_llama31_8b` and `gpt4o_teacher` needed the fuller five-shot `TEACHER_PROMPT` to produce valid YAML at all, since the short prompt never states the exact format rules; giving them the short prompt scored them at 0.000 on every metric, not because they reasoned badly, but because neither had seen the required shape before. So each model here is shown with its own best fair prompt, not one prompt held fixed across all four.

| Metric | untrained | this model (sys) | Llama-3.1-8B (Groq) | GPT-4o (teacher) |
|---|---|---|---|---|
| Format adherence | 0% | 98.9% | 100% | 100% |
| Target correctness | 0% | 85.3% | 93.7% | 98.9% |
| Exact match | 0% | 62.1% | 72.6% | 93.7% |

| Latency (seconds) | untrained | this model (sys) | Llama-3.1-8B (Groq) | GPT-4o (teacher) |
|---|---|---|---|---|
| Mean | 1.624 | 4.811 | 5.429 | 1.887 |
| Median | 1.267 | 4.901 | 5.391 | 1.160 |

This model does not win on accuracy or on latency against the two larger models, and that was never the bet. A 270-million-parameter model was never going to out-reason models 30 to 1000 times its size, and this benchmark ran the small models on a shared Colab T4 while Groq and GPT-4o run on production serving stacks built for speed at scale, so the latency gap is partly a research-notebook-versus-production-API gap, not a fair hardware comparison. The real case for this model is what the table does not show. It costs nothing per call once loaded, against an average of $0.000524 per call for the GPT-4o teacher, measured from real token counts on this benchmark's data and OpenAI's published pricing. No data leaves the machine it runs on, since neither the model-loading code nor the generation code makes a network call. And it keeps working with no network connection. I confirmed this directly by running generation with `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` set, which force a hard failure on any network call. It still produced a correct decomposition.

Two caveats on the table above. `target_correct` for GPT-4o is graded in part by an LLM judge that is itself GPT-4o, so its high score may partly reflect being judged by a copy of itself. And one of the 95 test questions is word for word one of `TEACHER_PROMPT`'s five worked examples, which gave `groq_llama31_8b` and `gpt4o_teacher` the answer to that one row for free.

**Does it actually reason, or just imitate the format?** To isolate how much of this model's result comes from fine-tuning versus from being told the format rules the short prompt never states, I gave the untrained base model the same full `TEACHER_PROMPT` that took Groq and GPT-4o from 0% to 100% on format adherence.

| Metric | untrained, short prompt | untrained, full TEACHER_PROMPT | this model (sys) |
|---|---|---|---|
| Format adherence | 0% | 21.1% | 98.9% |
| Target correctness | 0% | 12.6% | 85.3% |
| Exact match | 0% | 8.4% | 62.1% |

Even with the complete instructions, the untrained model could not use them for anything beyond copying fragments of the prompt's own worked examples. Reading the successful rows by hand, most were not real attempts at the question: one row's answer was lifted word for word from an unrelated worked example about Africa, and another borrowed the same example's phrasing for a question about a river. Most of the 95 calls came back nearly empty. That traced to the model's own configuration: 15 of its 18 layers use sliding-window attention limited to 512 tokens, and `TEACHER_PROMPT` plus a question runs 640 to 670 tokens, past that window, so most of the network never saw the beginning of the prompt, where the instructions live.

This model reaches 85.3% target correctness with a system prompt a fraction of `TEACHER_PROMPT`'s length, because it was trained on hundreds of examples of the actual task rather than told the rules once. That is the difference this project is built to measure: imitating the shape of an answer versus doing the task.

**What this model does not do.** It decomposes the first hop only. It does not retrieve documents, run the second hop, or answer the original question.

**License.** This is a fine-tune of `google/gemma-3-270m-it` and inherits the Gemma Terms of Use.

**Training data.** [Omokemi/hop-1-hotpotqa-decomposition](https://huggingface.co/datasets/Omokemi/hop-1-hotpotqa-decomposition)

**Code.** [github.com/D2himself/hop-specialist](https://github.com/D2himself/hop-specialist)
