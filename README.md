# WhatIsI

**A falsification-driven toy lab asking what computational role, if any, deserves the deictic word `I`.**

This repository does **not** test consciousness, qualia, personhood, or sentience. It asks a smaller functional question:

> If a learning system acts, receives consequences, persists through time, and repeatedly has to answer unrelated questions relative to the same causal agent, does it construct a persistent **deictic self-address** — and can language later bind `I` onto it?

The starting idea was that `I` might be a special transformer layer. The gates pushed the project somewhere else.

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
        -> useful scaffold; explicit provenance does NOT force self
        v
Gate 4B hidden causal actor
        -> transformer scene encoder + recurrent sidecar
           learns hidden causal body and reusable deictic state
        v
Gate 4C late I binding
        -> freeze Gate 4B first
           structured late lexical router binds self-produced I
           onto the old causal address
        v
Gate 4D generic binder attacker
        -> remove factorized address routing
           matched 256-label generic transformer FAILS the causal join
```

Current working hypothesis:

> **A self-address can be an ordinary persistent latent variable discovered because many computations share the same deictic binding. Language can later point to it, but making that binding cheaply reusable may require a useful compositional inductive bias.**

---

## Gate ledger

```text
Gate 0   local causal pointer                         PASS
Gate 1   same-size generic-memory attacker            SPECIAL-PRIMITIVE CLAIM FAILS
Gate 2   emergent deictic address, no self labels     PASS
Gate 3   frozen-core reuse on a new task              PASS
Gate 4A  long language life + Phi teacher             USEFUL SCAFFOLD, NO DEICTIC VERDICT
Gate 4B  hidden causal actor + transformer scene      PASS
Gate 4C  late lexical I -> frozen causal address      PASS
Gate 4D  generic unfactorized late binder             HYPOTHESIS FAILS
```

## Gates 0–3

Gate 0 showed that a tiny persistent action/consequence state can identify the causally controlled channel, persist through silence, ignore a stronger salience distractor, and rebind after transfer.

Gate 1 killed the claim that this required a special `I` primitive: a same-budget generic GRU performed better.

Gate 2 then removed explicit self labels. A generic recurrent model nevertheless formed a state from which the current causal owner was decodable and causally usable:

```text
query NMSE                         0.0910
linear hidden -> owner probe       0.9576
counterfactual state intervention  0.9744
```

Gate 3 froze that core and trained a tiny new head on a different self-relative task:

```text
frozen Gate-2 core       0.9979
same-size random core    0.7653
current sensory state    0.6630
```

This earned the limited interpretation **reusable deictic join key**.

---

## Gate 4A — Language Life

Status: **useful scaffold; not a deictic-identity gate.**

Gate 4A built a continuing tiny language/action learner with persistent state, replay consolidation, and an optional local Phi-3 teacher.

The 5000-turn Phi run learned language well, but analysis killed the stronger self interpretation:

- OBS/ACT provenance was explicitly supplied by source embeddings;
- `source-gap` was directly supervised;
- replay learned many phrase -> action mappings with zero persistent memory;
- most tasks did not require tracking a hidden causal body;
- `I` / `you` phrases could often be solved lexically.

See `docs/GATE4A_LANGUAGE_LIFE.md` and `docs/WINDOWS_PHI_TEACHER.md`.

---

## Gate 4B — Hidden Causal Actor: PASS

Each synthetic life contains four visible agents with stable body markers, mutable names/voices, and positions. Observation order changes every step. One body is controlled by the learner's motor stream, but no SELF/owner label is supplied.

Attacks include name/voice reassignment, silent windows, control transfer, and distractor motion.

Architecture:

```text
current relational scene
         |
         v
    tiny transformer
         |
 current entity tokens
         |
         +---------------------+
         |                     |
         v                     v
 scene-level roles       entity-key matching
                                ^
                                |
                     20-float recurrent sidecar
                                ^
                                |
                     motor-consequence feedback
```

Fresh seeds 3/4/5:

```text
actor consequence accuracy        0.923 / 0.919 / 0.928
zero-memory actor accuracy        0.232 / 0.254 / 0.232
post surface-swap accuracy        0.995 / 0.986 / 0.997
post transfer accuracy            0.999 / 1.000 / 1.000
silent self-body probe            0.944 / 0.944 / 0.834
held-out join NMSE                0.068 / 0.121 / 0.177
zero-memory join NMSE             0.927 / 0.937 / 0.935
counterfactual intervention       0.988 / 0.975 / 0.966
```

Safe conclusion:

> **A transformer scene encoder plus a generic persistent recurrent sidecar can learn a hidden causal-body binding that survives surface identity changes, persists through silence, rebinds after transfer, transfers to a new self-relative task, and causally controls that task under state intervention.**

See `docs/GATE4B_HIDDEN_ACTOR.md`.

---

## Gate 4C — Late `I` Binding: PASS

Gate 4B had a causal address and an ordinary role task (`I = speaker`, `YOU = addressee`), but those were not forced to be the same coordinate.

Gate 4C changes the developmental order:

1. train Gate 4B;
2. freeze it completely;
3. introduce only 256 late lexical labels.

A tiny factorized router is offered three existing referent sources:

```text
causal address
visible speaker
visible addressee
```

For self-produced `I`, no speaker/body identity is supplied.

Fresh final-evaluator seeds 3/4/5:

```text
self-produced I accuracy        0.9977 / 0.9942 / 0.9933
zero-state self I              0.2309 / 0.2289 / 0.2197
external I                     1.000 / 1.000 / 1.000
YOU                            1.000 / 1.000 / 1.000
counterfactual I state swap    0.9751 / 0.9479 / 0.9028
```

So a late lexical system can bind self-produced `I` onto a pre-existing causal address. Zeroing that address selectively destroys self-`I`; replacing it with another body's state moves the linguistic referent.

See `docs/GATE4C_LATE_I_BINDING.md`.

---

## Gate 4D — Generic Binder Attacker: HYPOTHESIS FAILS

Question:

> If we stop handing the learner `causal address / speaker / addressee` as separate routing options, does a generic architecture rediscover the same decomposition because it is useful?

Gate 4D freezes Gate 4B exactly as before, but replaces Gate 4C's factorized router with a generic two-layer transformer. It receives:

```text
4 current frozen agent representations
raw 20-float persistent state as ONE generic token
fused external/self-produced + I/YOU query
visible speaker/addressee IDs only when legitimately available
```

Its output is an unconstrained 8-way MLP over possible names. There is no precomputed causal-address distribution and no entity-pointer readout.

Matched Gate-4C late-data budget:

```text
256 balanced labels
800 optimizer updates
2 generic transformer layers
```

Fresh development runs through the final-code path:

```text
                              seed 3    seed 4    seed 5     mean
self-produced I               0.256     0.431     0.318      0.335
zero-memory self I            0.162     0.151     0.151      0.155
external I                    0.995     0.999     0.999      0.998
YOU                           0.999     1.000     1.000      1.000
counterfactual I state swap   0.155     0.170     0.190      0.171
```

The generic binder clearly trains — visible social deixis is essentially solved — but it does **not** rediscover the causal compositional join. Most importantly, counterfactual state replacement does not make the unchanged word `I` follow the injected body.

Safe conclusion:

> **Gate 4C's explicit factorization is a strong low-data inductive bias, not merely an interpretability convenience. The tested generic late transformer does not automatically rediscover that factorization at the same 256-label budget.**

This does **not** prove that generic architectures can never learn it. More data, end-to-end reuse pressure, recurrent KV/SSM memory, generic cross-attention, or other weaker biases remain live.

See `docs/GATE4D_GENERIC_BINDER.md`.

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
python experiments/gate4d_generic_binder.py
```

Smoke Gate 4D:

```bash
python experiments/gate4d_generic_binder.py --quick --seeds 3
```

Exploratory stress test, without changing the registered 256-label verdict:

```bash
python experiments/gate4d_generic_binder.py --seeds 3 --labels 4096 --updates 1500
```

Gate 4A Language Life remains available:

```bash
python experiments/gate4a_language_life.py --teacher scripted --steps 5000
```

Core tests:

```bash
python -m unittest discover -s tests -v
```

---

## What the project currently says

```text
slow learning / backprop
    learns how scenes are interpreted
    learns how persistent state is updated and used

one particular life / forward dynamics
    determines which represented entity occupies the causal role

late lexical learning
    can attach the word I to that coordinate

but
    generic downstream reuse is not automatically cheap;
    compositional structure / inductive bias matters
```

The next useful question is therefore:

> **What is the weakest architectural or training bias that makes the causal address compositional enough for generic downstream language to reuse?**

Candidates include generic cross-attention over current entities, multi-task relational reuse before language, recurrent KV/SSM memory, and local fast-weight causal addressing.

The stronger local-learning question also remains open:

> Can the reusable deictic address itself be acquired/revised by a local or fast-weight update rule while slow weights stay frozen?

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
