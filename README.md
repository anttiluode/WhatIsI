# WhatIsI

**A falsification-driven toy lab asking what computational role, if any, deserves the deictic word `I`.**

This repository does **not** test consciousness, qualia, personhood, or sentience. It asks a smaller functional question:

> If a learning system acts, receives consequences, persists through time, and repeatedly has to answer unrelated questions relative to the same causal agent, does it construct a persistent **deictic self-address** — and can language later bind `I` onto it?

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
        v
Gate 4C late I binding
        -> freeze Gate 4B first
           self-produced I binds onto the old causal address
```

Current hypothesis:

> **A self-address can be an ordinary persistent latent variable that learning discovers because many otherwise unrelated computations share the same deictic binding. Language can then learn that `I` is one pointer into that already-existing coordinate.**

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

The 5000-turn Phi run demonstrated useful language learning, but analysis killed the stronger self interpretation:

- provenance was explicitly supplied by source embeddings;
- `source-gap` was directly supervised;
- replay trained elementary phrase -> action mappings with zero persistent memory;
- the task rarely required knowing which represented body was causally "mine";
- `I` / `you` phrases could often be solved as ordinary lexical patterns.

So Gate 4A remains useful engineering, but **does not earn deictic identity**.

See `docs/GATE4A_LANGUAGE_LIFE.md` and `docs/WINDOWS_PHI_TEACHER.md`.

---

# Gate 4B — Hidden Causal Actor: PASS

Gate 4B makes the identity of the causal actor itself a hidden variable.

Each synthetic life contains four visible agents. Every body has a stable `shape` marker, mutable name/voice, and position. Observation order is randomized every step. One body is controlled by the learner's motor stream, but the model never receives a `SELF` or owner label.

The main target is consequence prediction:

> Given the current world and motor stream, which **currently visible name** will carry the motor consequence?

Attacks:

```text
surface swap     names + voices change, body stays
silent windows   no new motor consequence is available
control transfer motor stream suddenly controls another body
distractor move  another body moves too
```

Architecture:

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

The transformer handles current relational/language interpretation. A generic GRU sidecar carries continuity across moments. Zeroing the sidecar leaves current perception and language intact.

Fresh seeds 3 / 4 / 5:

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

Gate 4B left one important residual: its persistent causal address and its ordinary `I = speaker`, `YOU = addressee` role task were both useful, but they were not forced to be the same coordinate.

See `docs/GATE4B_HIDDEN_ACTOR.md`.

---

# Gate 4C — Late `I` Binding: PASS

Gate 4C attacks that residual by changing the developmental order.

1. Train Gate 4B exactly as before.
2. Freeze the entire Gate-4B causal core.
3. Only then introduce a tiny lexical router trained from **256 late labels**.

The router is offered three already-existing address sources:

```text
0  causal address       frozen Gate-4B persistent state -> current entity
1  visible speaker      ordinary social role
2  visible addressee    ordinary social role
```

The router itself sees only:

```text
speech source   external / self-produced
pronoun         I / YOU
```

For self-produced `I`, **no speaker name or body identity is supplied**. `self-produced` says only that the utterance came through the learner's output/efference channel; it does not identify which visible body currently occupies that causal role.

The learned routing becomes nearly one-hot:

```text
external + I    -> visible speaker
external + YOU  -> visible addressee
self + I        -> causal address
self + YOU      -> visible addressee
```

Fresh runs through the final committed evaluator, seeds 3 / 4 / 5:

```text
self-produced I accuracy        0.9977 / 0.9942 / 0.9933
zero-state self I              0.2309 / 0.2289 / 0.2197
external I                     1.000 / 1.000 / 1.000
external I with state zeroed   1.000 / 1.000 / 1.000
YOU                            1.000 / 1.000 / 1.000
YOU with state zeroed          1.000 / 1.000 / 1.000
counterfactual I state swap    0.9751 / 0.9479 / 0.9028
```

The decisive intervention holds fixed:

```text
visible world
lexical situation = SELF-PRODUCED + I
late lexical-router weights
slow Gate-4B weights
```

Only the old causal state is replaced by a centroid associated with another body currently visible in the same held-out world batch.

```text
same scene
same word I
same router

h_I(A) -> I refers to A
h_I(C) -> I refers to C
```

Safe conclusion:

> **A lexical first-person system introduced only after a causal self-address has already formed can learn to bind self-produced `I` onto that pre-existing address. Removing the causal state selectively destroys self-`I`; counterfactual state replacement moves the linguistic referent while the word and world remain fixed.**

This closes the specific Gate-4B residual, but Gate 4C is still deliberately factorized. The late learner is offered three candidate address channels; it does not prove that a completely unconstrained transformer would invent this routing structure unaided.

See `docs/GATE4C_LATE_I_BINDING.md`.

---

## Run

```bash
pip install -e '.[torch]'
python experiments/gate0_deictic_pointer.py
python experiments/gate1_generic_memory_attacker.py
python experiments/gate2_emergent_self_address.py
python experiments/gate3_reuse_factorization.py
python experiments/gate4b_hidden_actor.py
python experiments/gate4c_late_i_binding.py
```

Gate 4C single-seed smoke:

```bash
python experiments/gate4c_late_i_binding.py --quick --seeds 3
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

```text
slow learning / backprop
    learns how current scenes are interpreted
    learns how persistent state is updated and used

one particular life / forward dynamics
    determines which body currently occupies the causal role
    h_I(t) -> h_I(t+1)

late lexical learning
    can learn that self-produced I points to h_I
```

The slow weights learn **how to form and use a deictic address**. A particular stream of action and consequence determines **which entity that address refers to here and now**. Gate 4C shows that language can later learn **which word points to it**.

The stronger open problems remain:

- remove the explicit factorized entity/address readouts and compare generic equal-budget architectures;
- test natural varied first-person language rather than the symbolic late lexical router;
- acquire/revise the reusable deictic state by local/test-time plasticity while slow weights stay frozen.

---

## Working definition

For this repository, `I` provisionally means:

> **A persistent, reusable deictic state that binds ongoing computation to the entity whose actions have the relevant causal consequences; linguistic `I` can be a learned pointer to that state.**

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
