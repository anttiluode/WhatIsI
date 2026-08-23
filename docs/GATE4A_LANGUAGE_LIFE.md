# Gate 4A — Language life with causal provenance

Status: **exploratory build, not yet a passed gate**.

## Question

Can a small transformer begin to form a useful persistent deictic state when language arrives through two causally different streams?

```text
OBS  language generated outside the learner
ACT  language actually emitted by the learner
```

The lexical bytes may be identical. Only `ACT` is wired to change the symbolic world. The outside world sometimes repeats the learner's exact action phrase as `OBS`; that repetition is deliberately non-causal.

This attacks the easy shortcut:

> "self-generated language is recognizable because it contains different words."

It does not. The distinction is provenance and consequence.

## Architecture

A small transformer reads byte tokens plus a separate source/provenance embedding. A 32-float recurrent life-state is projected in as a memory token. The state is updated by a GRU cell from current representation, emitted action and observed consequence.

Training uses short truncated unrolls (default 8 turns). Backpropagation therefore trains the machinery that *updates* persistent state without backpropagating through the entire synthetic lifetime.

The learner chooses one of 13 simple language actions such as `say red`, `touch blue`, `take green`, `move left`, and `wait`. The action is executed before feedback exists.

A teacher supplies outside language. The default scripted teacher makes the experiment reproducible. Optional Phi-3 via GGUF/llama.cpp or Ollama periodically expands the paraphrase bank; the large model is **not** allowed to label self/identity or inspect the student's hidden state.

## Developmental curriculum

Before `deixis_after` (default 2000 turns), the teacher bank avoids first- and second-person pronouns. The student must first learn the causal difference between language that arrives and language it emits.

After that point, teacher paraphrases may contain `I` and `you`. This creates a sharper later question:

> do deictic words bind onto a pre-existing causal coordinate, or does the network simply memorize their surface statistics?

Use `--deixis-after -1` to keep pronouns disabled for an entire run.

## Why Phi-3 Mini

Microsoft's official Phi-3 Mini 4K GGUF is a 3.8B-parameter model. Its Q4_K_M file is about 2.2 GB, making it a reasonable frozen local language teacher for this experiment. It is MIT licensed.

Recommended teacher file:

```text
microsoft/Phi-3-mini-4k-instruct-gguf
Phi-3-mini-4k-instruct-q4.gguf
```

The teacher is replaceable. A result that only occurs with Phi-3 wording is not a deictic result.

## Live measurements

The runner displays:

- action imitation accuracy;
- online training loss;
- persistent memory norm;
- `source-gap`: predicted world-changing probability for the **same phrase** supplied as `ACT` versus `OBS`.

A positive source-gap is necessary but nowhere near sufficient for a self-address claim. It only says the model has learned causal language provenance.

## Running

Scripted/reproducible first:

```bash
pip install -e '.[torch]'
python experiments/gate4a_language_life.py --teacher scripted --steps 5000
```

Download the small Phi teacher:

```bash
pip install -e '.[teacher]'
python experiments/download_phi_teacher.py
```

Direct GGUF teacher:

```bash
python experiments/gate4a_language_life.py \
  --teacher llama \
  --model models/Phi-3-mini-4k-instruct-q4.gguf \
  --steps 5000
```

Or, if a local Ollama server already has Phi-3:

```bash
python experiments/gate4a_language_life.py --teacher ollama --ollama-model phi3:mini --steps 5000
```

GUI:

```bash
python experiments/run_language_gui.py --teacher scripted
```

Add `--resume` to continue from `runs/language_life.pt`.

## Next falsifier

After a long run, freeze the student and train a tiny held-out probe/task that requires a delayed first-person binding but never receives source labels directly. Then intervene on the persistent state, as in Gates 2–3.

If an equal-sized ordinary context-only transformer or generic memory solves the same held-out task just as well, the persistent deictic interpretation is demoted.
