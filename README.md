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
Gate 4A language life
        -> hear language / emit language / receive consequences
           with a persistent transformer state
```

The current hypothesis is deliberately modest:

> **A self-address may be an ordinary persistent latent variable that learning discovers because many otherwise unrelated predictions share the same deictic binding.**

---

## Gates 0–3: what is already earned

### Gate 0 — local online deictic state: PASS

Four candidate sensorimotor channels are present. Only one is causally controlled by the agent's motor command. A four-float state updates from local action/consequence correlation, survives silent intervals, ignores a more energetic distractor, and rebinds after agency transfer.

Fresh `1000..1039`:

```text
causal-pointer accuracy                 1.000
silent-window accuracy                  1.000
salience-memory attacker                0.253
median recovery after transfer          14 active steps
```

This establishes that a persistent causal binding can be acquired online without lifetime backpropagation.

### Gate 1 — generic-memory attacker: SPECIAL-PRIMITIVE CLAIM FAILS

Replace the hand-written pointer with a generic GRU using the **same four persistent floats**. It performs better overall and adapts much faster after ownership changes.

Hard OOD:

```text
                         pointer      generic GRU
all                        0.844          0.901
silent                     0.939          0.915
post-transfer              0.547          0.810
```

So naming a state `I` does not make it a new primitive.

### Gate 2 — emergent deictic address without self labels: PASS

An 8-float generic GRU receives sensorimotor history and fresh random values attached to four candidate entities. Its only training target is:

```text
return the fresh value attached to the currently causally controlled entity
```

The owner/self label is never provided.

Fresh `401..403`:

```text
query NMSE                              0.0910
linear hidden-state -> owner probe      0.9576
```

Causal intervention matters more than the probe: replace the hidden state with the mean state associated with another owner while keeping current query values fixed.

```text
output follows injected owner           0.9744
```

The recurrent state contains a causally used deictic variable even though the objective never names one.

### Gate 3 — reuse / factorization: PASS

Freeze the Gate-2 recurrent core. Invent a different binary self-relative task. Train only a tiny new head from 256 task labels; still never provide owner labels. Evaluate only in silent periods.

```text
frozen Gate-2 core          0.9979
same-size random core       0.7653
current sensory state       0.6630
oracle                      1.0000
```

This is the strongest reason so far to call the latent a **join key** rather than merely a task-specific code.

---

# Gate 4A — Language Life

Status: **built and runnable; exploratory, not yet a passed gate.**

The new question is whether the same idea can arise when the system learns from a continuing language/action stream.

The key asymmetry is causal, not lexical:

```text
OBS language
    generated outside the student

ACT language
    actually emitted by the student
    -> changes the world
    -> consequence returns later
```

The same bytes can occur in either stream.

The outside world deliberately repeats some student action phrases as `OBS`. Those repeated words do **not** execute an action. Therefore the learner cannot solve provenance by noticing special vocabulary.

## Student

`whatisi/language_life.py` contains a small byte-level transformer:

```text
outside utterance OBS
        |
        v
byte tokens + provenance embedding
        |
        v
small transformer <----- persistent 32-float life state
        |
        +---- action phrase
        |       |
        |       v
        |   symbolic world
        |       |
        |       v
        +---- consequence / feedback
                |
                v
        recurrent life-state update
```

The student currently emits one of 13 simple language/actions such as:

```text
red
blue
touch red
take green
left
right
...
```

The internal semantic action is hidden from the input stream. The emitted surface phrase is what enters history.

## Backpropagation versus living

Gate 4A makes the separation explicit.

Backpropagation runs through short truncated unrolls (default 8 turns) and trains **how the machinery updates persistent state**.

It does not backpropagate through the entire synthetic lifetime.

```text
slow/developmental training
    learn transformer + update rule

ongoing life
    h(t) -> h(t+1)
    because action and consequence occurred
```

This is not yet the local/test-time plasticity result sought by the original Gate 4. It is a bridge experiment.

## Teacher

The teacher is deliberately outside the scientific mechanism. It only supplies simple language/paraphrases.

Three teacher backends exist:

```text
scripted     deterministic baseline
llama        local GGUF via llama-cpp-python
ollama       local Ollama model
```

A teacher may never inspect student hidden state or label `self`/identity.

### Recommended small local LLM teacher

The intended default is Microsoft's **Phi-3 Mini 4K Instruct** GGUF:

```text
microsoft/Phi-3-mini-4k-instruct-gguf
Phi-3-mini-4k-instruct-q4.gguf
```

The official Q4_K_M file is about 2.2 GB. The teacher periodically expands a cached paraphrase bank; it does not need to run for every student token.

## Developmental curriculum

Before turn 2000 by default, teacher language avoids first- and second-person pronouns. The learner first has to discover causal language provenance.

After that point `I`/`you` phrasing may enter the outside language stream.

This gives us a later falsifiable question:

> **Do deictic words bind onto an already useful causal coordinate, or does the network merely memorize their surface statistics?**

Use `--deixis-after -1` to keep pronouns disabled indefinitely.

## Live measurements

The console and GUI show:

- action-learning accuracy;
- online loss;
- persistent memory norm;
- `source-gap` — predicted probability of a world-changing consequence for the **same phrase** presented as `ACT` versus `OBS`.

A positive source-gap is only a prerequisite. It is not evidence of a self-address by itself.

The real later test is the Gate-2/3 style test: freeze the student, invent a held-out delayed self-relative task, probe the persistent state, and intervene on it.

---

## Run

Core gates:

```bash
pip install -e '.[torch]'
python experiments/gate0_deictic_pointer.py
python experiments/gate1_generic_memory_attacker.py
python experiments/gate2_emergent_self_address.py
python experiments/gate3_reuse_factorization.py
```

Language Life with the reproducible teacher:

```bash
python experiments/gate4a_language_life.py --teacher scripted --steps 5000
```

GUI:

```bash
python experiments/run_language_gui.py --teacher scripted
```

Download Phi-3 Mini Q4:

```bash
pip install -e '.[teacher]'
python experiments/download_phi_teacher.py
```

Then:

```bash
python experiments/gate4a_language_life.py \
  --teacher llama \
  --model models/Phi-3-mini-4k-instruct-q4.gguf \
  --steps 5000
```

Or use a local Ollama server:

```bash
python experiments/gate4a_language_life.py \
  --teacher ollama \
  --ollama-model phi3:mini \
  --steps 5000
```

Runs checkpoint to `runs/language_life.pt`. Add `--resume` to continue.

Tests:

```bash
python -m unittest discover -s tests -v
```

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
    how is the model expressed through language and autobiography?
```

The experimental evidence currently supports only the first object.

## Repo rule

> **Do not call a state `I` because we named it that. Make the system need a deictic variable, let it learn whatever it learns, then probe, transfer, intervene, and attack the obvious shortcuts.**
