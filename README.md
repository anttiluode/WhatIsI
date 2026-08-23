# WhatIsI

**A falsification-driven toy lab asking what computational role, if any, deserves the deictic word `I`.**

This repository does **not** test consciousness, qualia, personhood, or sentience. It asks a smaller functional question:

> If a learning system acts, receives consequences, persists through time, and repeatedly has to answer unrelated questions relative to the same causal agent, does it construct a persistent **deictic self-address**?

The starting idea was that `I` might be a special transformer layer. The gates have pushed the project somewhere else.

```text
"I is a special layer"
        |
        v
Gate 0  local causal pointer works
        |
        v
Gate 1  generic recurrence works better
        |      -> special primitive claim dies
        v
Gate 2  no self labels
        -> deictic address emerges
        v
Gate 3  freeze core, invent new task
        -> latent transfers as reusable join key
        v
Gate 4A language life
        -> useful scaffold, but explicit provenance did NOT force self
        v
Gate 4B hidden causal actor
        -> transformer scene encoder + recurrent sidecar
           learns hidden causal body and reusable deictic state
```

Current hypothesis:

> **A self-address can be an ordinary persistent latent variable that learning discovers because many otherwise unrelated computations share the same deictic binding.**

---

## Gates 0–3

### Gate 0 — local online deictic state: PASS

A four-float action/consequence correlation state identifies the causally controlled channel, survives silent intervals, ignores a stronger salience distractor, and rebinds after agency transfer.

```text
accuracy                 1.000
silent accuracy          1.000
salience attacker        0.253
median transfer recovery 14 active steps
```

### Gate 1 — generic-memory attacker: SPECIAL-PRIMITIVE CLAIM FAILS

A generic GRU with the **same four persistent floats** performs better overall and adapts faster after ownership transfer.

```text
hard OOD all       pointer 0.844   generic 0.901
hard OOD transfer  pointer 0.547   generic 0.810
```

Naming a state `I` does not make it a new primitive.

### Gate 2 — emergent deictic address without self labels: PASS

An 8-float generic GRU is never given an owner/self label. It must answer fresh queries relative to whichever entity is currently causally controlled.

```text
query NMSE                         0.0910
linear hidden -> owner probe       0.9576
counterfactual state intervention  0.9744
```

Replacing hidden state with a state associated with another owner makes the model answer relative to that other owner.

### Gate 3 — reuse / factorization: PASS

Freeze the Gate-2 recurrent core. Train only a tiny new head on a different self-relative task from 256 labels.

```text
frozen Gate-2 core       0.9979
same-size random core    0.7653
current sensory state    0.6630
```

This is the strongest reason to call the latent a reusable **join key** rather than merely a task-specific code.

---

# Gate 4A — Language Life

Status: **useful scaffold; not a deictic-identity gate.**

Gate 4A built a continuing language/action learner with a tiny transformer, persistent state, replay consolidation, and optional Phi-3 teacher.

It demonstrated that a small student can learn from a long outside-language stream and distinguish language arriving through separate `OBS` and `ACT` channels.

But the analysis killed the stronger interpretation:

- provenance was explicitly supplied by source embeddings;
- the auxiliary `source-gap` objective directly supervised ACT-vs-OBS distinction;
- replay trained elementary phrase -> action mappings with zero persistent memory;
- the task rarely required knowing which represented body was causally "mine";
- `I` / `you` phrases could often be solved as ordinary lexical patterns.

So Gate 4A remains useful engineering, but **does not earn deictic identity**.

The Phi teacher path remains available through Ollama or GGUF. See `docs/GATE4A_LANGUAGE_LIFE.md` and `docs/WINDOWS_PHI_TEACHER.md`.

---

# Gate 4B — Hidden Causal Actor: PASS

Gate 4B makes the identity of the causal actor itself a hidden variable.

Each synthetic life contains four visible agents. Every body has:

- a stable `shape` marker,
- a mutable name,
- a mutable voice,
- a position.

Observation order is randomized every step. One body is controlled by the student's motor stream, but the model never receives a `SELF` or owner label.

The main target is consequence prediction:

> Given the current world and motor stream, which **currently visible name** will carry the motor consequence?

The body/name distinction matters because names and voices are reassigned mid-life while the controlled body remains the same.

Additional attackers:

```text
surface swap     names + voices change, body stays
silent windows   no new motor consequence is available
control transfer motor stream suddenly controls another body
distractor move  another body moves too
```

## Architecture

Gate 4B ended up with a cleaner separation than Gate 4A:

```text
current relational scene + language roles
                 |
                 v
          tiny transformer
                 |
        momentary entity tokens
                 |
                 +------------------+
                 |                  |
                 v                  v
          I / YOU task       entity-key lookup
                                    ^
                                    |
                         persistent 20-float state
                                    ^
                                    |
                         motor consequence feedback
```

The transformer handles current relational/language interpretation. A generic GRU sidecar carries continuity across moments. Zeroing the sidecar leaves current perception and language intact, giving a clean ablation.

## Deictic language

Every step also chooses an arbitrary speaker and addressee. The model receives their names plus either `I` or `YOU` and must recover the referred agent's position.

```text
I   -> current speaker
YOU -> current addressee
```

The referent changes constantly, so no fixed entity can be memorized as `I`.

## Held-out reuse and intervention

After training, freeze the core.

During silent periods attach fresh random values to all possible bodies. Train only a tiny new selector from **256 task labels** to return the value belonging to the current causal body. The selector never receives owner labels.

Then replace persistent state with a centroid associated with another body while keeping the current query values fixed.

Fresh local runs, seeds 3 / 4 / 5:

```text
actor consequence accuracy        0.923 / 0.919 / 0.928
zero-memory actor accuracy        0.232 / 0.254 / 0.232

post surface-swap accuracy        0.995 / 0.986 / 0.997
post transfer accuracy            0.999 / 1.000 / 1.000

I/YOU role accuracy               0.895 / 0.945 / 0.912
silent self-body linear probe     0.944 / 0.944 / 0.834

held-out join NMSE                0.068 / 0.121 / 0.177
zero-memory join NMSE             0.927 / 0.937 / 0.935

counterfactual state intervention 0.988 / 0.975 / 0.966
```

All three fresh seeds pass.

Safe conclusion:

> **A small transformer for current relational/language interpretation plus a generic persistent recurrent sidecar can learn a hidden causal-body binding that survives surface identity swaps, rebinds after control transfer, remains decodable through silent intervals, transfers to a new self-relative task, and causally controls that task under state intervention.**

This is the first transformer-side result that reconnects cleanly to Gates 2–3.

It still does **not** show that a GRU sidecar is special, that this placement beats persistent tokens/KV/fast weights, or that the system is conscious.

See `docs/GATE4B_HIDDEN_ACTOR.md`.

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

Gate 4B:

```bash
python experiments/gate4b_hidden_actor.py
```

Quick smoke run:

```bash
python experiments/gate4b_hidden_actor.py --quick --seeds 3
```

Gate 4A Language Life remains available:

```bash
python experiments/gate4a_language_life.py --teacher scripted --steps 5000
```

Tests:

```bash
python -m unittest discover -s tests -v
```

---

## What backpropagation means here

The project now has a cleaner answer to the original question.

```text
slow learning / backprop
    learns how current scenes are interpreted
    learns how persistent state is updated and used

one particular life / forward dynamics
    determines which body currently occupies the causal role
    h_I(t) -> h_I(t+1)
```

The slow weights learn **how to form and use a deictic address**. A particular stream of action and consequence determines **which entity that address refers to here and now**.

The stronger open problem remains:

> Can the reusable deictic state itself be acquired and revised by a local/test-time plasticity rule while slow weights stay frozen?

---

## Working definition

For this repository, `I` provisionally means:

> **A persistent, reusable deictic state that binds ongoing computation to the entity whose actions have the relevant causal consequences.**

That is not yet a full self-model.

```text
self-address
    which entity is the causal first-person referent?

self-model
    what can that entity do, remember, predict, prefer?

self-narrative
    how is the model expressed through language and autobiography?
```

## Repo rule

> **Do not call a state `I` because we named it that. Make the system need a deictic variable, let it learn whatever it learns, then probe, transfer, intervene, and attack the obvious shortcuts.**
