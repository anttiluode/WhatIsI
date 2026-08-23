# WhatIsI

**A falsification-driven toy lab asking what computational role, if any, deserves the deictic word `I`.**

This repository does **not** test consciousness, qualia, personhood, or sentience. It asks a smaller functional question:

> If a learning system acts, receives consequences, persists through time, and repeatedly has to answer unrelated questions relative to the same causal agent, does it construct a persistent **deictic self-address**?

The starting intuition was that `I` might be a special layer in a transformer. The first gates killed that formulation.

```text
"I is a special layer"
        |
        v
Gate 0  local causal pointer works
        |
        v
Gate 1  generic 4-float GRU works better
        |      -> special primitive claim dies
        v
Gate 2  generic recurrence, no self labels
        -> deictic address emerges anyway
        v
Gate 3  freeze core, invent new task
        -> latent transfers as reusable join key
        v
Gate 4A Language Life
        -> outside language / emitted language / consequences
           with persistent transformer state + replay consolidation
```

Current hypothesis:

> **A self-address may be an ordinary persistent latent variable that learning discovers because many otherwise unrelated predictions share the same deictic binding.**

---

## What Gates 0–3 earned

### Gate 0 — local online deictic state: PASS

A four-float action/consequence correlation state identifies the causally controlled channel, survives silent intervals, ignores a deliberately more energetic distractor, and rebinds after agency transfer.

Fresh `1000..1039`:

```text
causal-pointer accuracy                 1.000
silent-window accuracy                  1.000
salience-memory attacker                0.253
median recovery after transfer          14 active steps
```

### Gate 1 — generic-memory attacker: SPECIAL-PRIMITIVE CLAIM FAILS

A generic GRU with the **same four persistent floats** learns the same binding and performs better overall, especially after ownership transfer.

Hard OOD:

```text
                         pointer      generic GRU
all                        0.844          0.901
silent                     0.939          0.915
post-transfer              0.547          0.810
```

Naming a state `I` does not make it a new primitive.

### Gate 2 — emergent deictic address without self labels: PASS

An 8-float generic GRU receives sensorimotor history plus fresh random values attached to four candidate entities. Its only training target is:

```text
return the fresh value attached to the currently causally controlled entity
```

The owner/self label is never supplied.

Fresh `401..403`:

```text
query NMSE                              0.0910
linear hidden-state -> owner probe      0.9576
counterfactual state intervention       0.9744
```

Replacing hidden state with a state associated with another owner makes the model answer relative to that other owner.

### Gate 3 — reuse / factorization: PASS

Freeze the Gate-2 recurrent core. Invent a different binary self-relative task. Train only a tiny new head from 256 labels and still never provide owner labels.

```text
frozen Gate-2 core          0.9979
same-size random core       0.7653
current sensory state       0.6630
oracle                      1.0000
```

This is the strongest reason so far to call the latent a reusable **join key** rather than a task-specific code.

---

# Gate 4A — Language Life

Status: **built and runnable; exploratory, not yet a passed gate.**

The new experiment asks whether causal/deictic structure can begin to form from a continuing language/action stream.

The key distinction is causal, not lexical:

```text
OBS language
    arrived from outside the student

ACT language
    was actually emitted by the student
    -> enters the world
    -> consequences return later
```

The exact same phrase can occur in either stream. The outside world deliberately repeats some student action phrases as `OBS`; those repeated words do **not** execute an action.

So the student cannot identify self-generated language merely from vocabulary.

## Student

The current student is intentionally small:

```text
stable hashed word vocabulary           2048
transformer width                         64
transformer blocks                         2
attention heads                            4
persistent life-state                     24 floats
action vocabulary                         13
```

The phrase `touch red` therefore has identical word-token IDs whether heard or emitted. Provenance is supplied separately as `OBS`, `ACT`, `FEEDBACK`, or `WORLD`.

The current world description and current outside utterance choose the next action **together with the persistent state**. Older transcript text is not directly fed to the action policy. If the past matters, the life-state has to carry it.

```text
outside utterance OBS
       +
current world
       |
       v
small transformer <------ persistent h(t)
       |
       v
student ACT phrase
       |
       v
symbolic world
       |
       v
feedback / consequence
       |
       v
persistent h(t+1)
```

The internal semantic target is never shown as an input. The student hears only the teacher utterance, emits its own phrase, and then receives consequences.

## Backpropagation, replay, and living

The implementation ended up with two useful timescales:

```text
FAST / LIVING
    persistent state updates every turn by forward dynamics

SLOW / CONSOLIDATION
    recent heard-language episodes are replayed periodically
    transformer weights receive gradient updates
```

Short truncated unrolls (default 8 turns) train the state-update machinery. Every 64 lived turns, the persistent state is detached and recent teacher episodes receive 64 replay consolidation updates.

Replay was not added as decoration. In development, one-pass exposure made the tiny transformer waste its capacity relearning the elementary phrase mapping. Periodic replay lets ordinary language mappings consolidate while the persistent state remains the continuing trajectory.

This is close to the architectural distinction motivating the repo:

```text
backprop/replay learns how the machinery works
forward life changes which state this individual is in
```

It still does **not** prove the stronger Gate-4 goal: deictic state acquired through fully local/test-time plasticity while slow weights stay frozen.

## Teacher

The teacher is deliberately outside the scientific mechanism. It only generates simple outside-language paraphrases.

Backends:

```text
scripted     reproducible baseline
llama        local GGUF through llama-cpp-python
ollama       installed local Ollama model
```

The teacher cannot inspect hidden state, label `self`, or perform the later deictic probe.

### Recommended small teacher

The intended local LLM is Microsoft's **Phi-3 Mini 4K Instruct** GGUF:

```text
microsoft/Phi-3-mini-4k-instruct-gguf
Phi-3-mini-4k-instruct-q4.gguf
```

It is a 3.8B-parameter model; the official Q4 file is about 2.2 GB. The teacher only refreshes a cached paraphrase bank occasionally, so it does not need to run for every student token.

## Developmental language curriculum

Before `deixis_after` (default turn 2000), teacher language avoids first- and second-person pronouns.

The student therefore first has to learn the causal difference between outside language and its own emitted language.

Afterward, teacher paraphrases may contain `I` and `you`.

That gives a later falsifiable question:

> **Do deictic words bind onto causal structure that was already useful before those words appeared, or do the words merely create a surface shortcut?**

Use `--deixis-after -1` to keep pronouns disabled indefinitely.

## What you can watch

The console and GUI show:

- rolling 100-turn instruction accuracy;
- whole-life accuracy;
- training loss;
- persistent-state norm;
- `source-gap`: causal-authority prediction for the **same phrase** as `ACT` minus the same phrase as `OBS`.

A positive source-gap means only that causal language provenance has been learned. It is **not** a self-address verdict.

### Development sanity run

One scripted seed was run for 512 turns only to verify that the loop learns rather than displaying random behavior forever:

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

This is **not a registered gate**. It proves only that the experimental plumbing is usable.

---

## Run it

Install the student and begin with the reproducible teacher:

```bash
pip install -e '.[torch]'
python experiments/run_language_gui.py --teacher scripted
```

Headless:

```bash
python experiments/gate4a_language_life.py --teacher scripted --steps 5000
```

Download Phi-3 Mini Q4:

```bash
pip install -e '.[teacher]'
python experiments/download_phi_teacher.py
```

GUI with Phi:

```bash
python experiments/run_language_gui.py \
  --teacher llama \
  --model models/Phi-3-mini-4k-instruct-q4.gguf
```

Headless:

```bash
python experiments/gate4a_language_life.py \
  --teacher llama \
  --model models/Phi-3-mini-4k-instruct-q4.gguf \
  --steps 5000
```

Or use an already installed Ollama model:

```bash
python experiments/run_language_gui.py --teacher ollama --ollama-model phi3:mini
```

Checkpoints go to `runs/language_life.pt`. Add `--resume` to continue the same run. Logs and the accumulated teacher paraphrase bank are also kept under `runs/`.

Core tests:

```bash
python -m unittest discover -s tests -v
```

---

## The actual Gate 4A falsifier

Interesting-looking dialogue earns nothing by itself.

After a long run, freeze the student and create a **new delayed self-relative task that never receives source labels**. Compare:

```text
persistent life-state
context-only matched transformer
random persistent state
state-shuffled / state-intervened model
```

Then alter the persistent state while holding current words and world fixed.

Only if a new task can reuse the state, and behavior follows causal intervention on that state, does Language Life reconnect to the earned Gate-2/3 result.

---

## Working definition

For this repository, `I` provisionally means:

> **A persistent, reusable deictic state that binds ongoing computation to the entity whose actions have the relevant causal consequences.**

That is not yet a complete self-model.

```text
self-address
    which entity is the causal first-person referent?

self-model
    what can that entity do, remember, predict, prefer?

self-narrative
    how is that model expressed through language and autobiography?
```

## Repo rule

> **Do not call a state `I` because we named it that. Make the system need a deictic variable, let it learn whatever it learns, then probe, transfer, intervene, and attack the obvious shortcuts.**
