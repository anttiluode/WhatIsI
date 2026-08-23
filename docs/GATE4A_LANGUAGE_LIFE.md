# Gate 4A — Language Life with causal provenance

Status: **built and runnable; exploratory, not yet a passed gate**.

## Question

Can a small transformer begin to form a useful persistent deictic state when language arrives through two causally different streams?

```text
OBS  language generated outside the learner
ACT  language actually emitted by the learner
```

The lexical tokens can be identical. Only `ACT` is wired to change the symbolic world. The outside world sometimes repeats the learner's exact action phrase as `OBS`; that repetition is deliberately non-causal.

This attacks the easy shortcut:

> "self-generated language is recognizable because it contains different words."

It does not. The distinction is provenance and consequence.

## Architecture

The current student is intentionally small:

```text
hashed word-token vocabulary      2048
model width                         64
transformer blocks                   2
attention heads                       4
persistent life-state               24 floats
simple action vocabulary             13
```

Stable hashed word tokens are used rather than a learned tokenizer. Provenance is a separate embedding, so the phrase `touch red` receives identical token IDs as `OBS` and `ACT`; only the source channel differs.

The current world description and current outside utterance choose the action together with the persistent state. Older transcript text is deliberately excluded from the policy input. If the past matters, the persistent state has to carry it.

After the learner emits a phrase, the symbolic world executes the corresponding action and only then returns feedback. A GRU cell updates the persistent state from post-consequence representation, emitted action, and consequence bit.

## Slow learning, replay, and living

Training uses two timescales.

```text
ongoing life
    persistent state changes every turn by forward dynamics

slow consolidation
    lived teacher episodes are replayed periodically
    transformer weights receive gradient updates
```

The default student also uses short truncated unrolls of 8 turns. Every 64 lived turns it detaches the persistent state and performs 64 replay consolidation updates from recent heard-language episodes.

The replay buffer was added for a concrete reason: one-pass language exposure made the tiny model spend most of its capacity relearning the elementary phrase mapping. Replay lets the slow language machinery consolidate repeated episodes while the persistent state continues to represent the current life trajectory.

This does **not** yet answer the stronger original Gate 4 question: can a reusable deictic state itself be acquired by a local/test-time plasticity rule while slow weights remain frozen?

## Teacher

The teacher is outside the scientific mechanism. It only supplies simple outside-language paraphrases.

Three backends are supported:

```text
scripted     deterministic/reproducible baseline
llama        local GGUF through llama-cpp-python
ollama       any suitable local Ollama model
```

The LLM teacher may not inspect the student's hidden state, label `self`, or solve the downstream probe. It periodically expands a cached paraphrase bank rather than participating in every student token.

### Recommended small local teacher

Microsoft's official **Phi-3 Mini 4K Instruct** GGUF is a 3.8B-parameter model. The recommended Q4 file is roughly 2.2 GB:

```text
microsoft/Phi-3-mini-4k-instruct-gguf
Phi-3-mini-4k-instruct-q4.gguf
```

The teacher is replaceable. A result that only appears with Phi wording is not a deictic result.

## Developmental curriculum

Before `deixis_after` (default 2000 turns), teacher language avoids first- and second-person pronouns. The student first has to learn the causal difference between language that arrives and language it emits.

After that point, teacher paraphrases may contain `I` and `you`.

The later test is therefore:

> **Do deictic words bind onto a causal coordinate that was already useful before those words appeared, or do they merely create a surface-language shortcut?**

Use `--deixis-after -1` to keep pronouns disabled indefinitely.

## Live measurements

The console and GUI display:

- rolling 100-turn instruction accuracy;
- whole-life accuracy;
- current training loss;
- persistent-state norm;
- `source-gap` = predicted causal authority for the **same phrase** presented as `ACT` minus `OBS`.

A positive source-gap is only evidence that causal language provenance was learned. It is **not** by itself evidence of a self-address.

## Development sanity run

A single scripted seed was run for 512 turns only to verify that the user-facing loop actually learns rather than printing random behavior forever. This is **not a registered gate and has no scientific verdict**.

```text
turn    recent accuracy    source-gap
  64         0.172          -0.050
 128         0.080          +0.018
 192         0.180          +0.117
 256         0.330          +0.203
 320         0.500          +0.382
 384         0.760          +0.450
 448         0.880          +0.508
 512         0.930          +0.898
```

This establishes only that the plumbing is usable: periodic replay can teach the simple language mapping and the model can distinguish causal `ACT` from non-causal identical `OBS` phrases.

## Running

Start with the reproducible teacher:

```bash
pip install -e '.[torch]'
python experiments/gate4a_language_life.py --teacher scripted --steps 5000
```

GUI:

```bash
python experiments/run_language_gui.py --teacher scripted
```

Download the Phi teacher:

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

GUI with Phi:

```bash
python experiments/run_language_gui.py \
  --teacher llama \
  --model models/Phi-3-mini-4k-instruct-q4.gguf
```

Or with an already installed Ollama model:

```bash
python experiments/gate4a_language_life.py --teacher ollama --ollama-model phi3:mini --steps 5000
```

Add `--resume` to continue from `runs/language_life.pt`.

## Next falsifier

After a long run, freeze the student and invent a delayed self-relative task that never receives source labels. Compare:

```text
persistent life-state
context-only matched transformer
random persistent state
state-shuffled / state-intervened model
```

Then intervene on the persistent state while keeping the current words and world fixed, as in Gates 2–3.

If an equal-sized generic/context-only system solves the held-out task just as well, the deictic interpretation is demoted.
