# WhatIsI — current handoff

Date: 2026-08-23

## Restart from here

Do **not** restart from the idea that `I` is a special transformer layer.

The current earned object is narrower and stronger:

> **A generic persistent state can act as a reusable deictic address for whichever represented entity currently occupies the learner's causal role. A transformer can use that state to join momentary relational/language interpretation to a continuing first-person referent.**

Functional result only; not a consciousness claim.

## Gates 0–3

```text
Gate 0  local 4-float causal pointer                 PASS
Gate 1  same-size generic GRU attacker               SPECIAL-PRIMITIVE CLAIM FAILS
Gate 2  generic recurrence, no self labels           PASS
Gate 3  freeze core, invent new self-relative task   PASS
```

Key Gate-2/3 result: a recurrent latent can become a causally used, reusable deictic join key without an explicit self label.

## Gate 4A — Language Life

Status: **useful scaffold; not a deictic-identity gate.**

It built a long-running tiny language learner with:

- outside language,
- self-emitted language,
- symbolic consequences,
- 24-float persistent state,
- replay consolidation,
- optional Phi-3 teacher.

The 5000-turn Phi run learned language well, but analysis killed the stronger interpretation:

- OBS/ACT provenance was explicitly supplied by source embeddings;
- `source-gap` was directly supervised;
- replay taught many phrase->action mappings with zero memory;
- most language tasks did not require tracking a hidden causal body;
- `I`/`you` phrases could often be solved lexically.

Do not use Gate 4A as evidence for deictic identity.

## Gate 4B — Hidden Causal Actor: PASS

This is now the main transformer-side result.

### World

Each life has four visible agents with:

```text
stable body marker = shape
mutable name
mutable voice
position
```

Observation order is randomized every step.

One body is controlled by the motor stream, but the model is never given a SELF/owner label.

It must predict which **current visible name** will carry the consequence of its motor action.

Mid-life attackers:

```text
surface swap     names and voices reassigned, controlled body unchanged
silent windows   no motor consequence available
control transfer motor stream suddenly routed to another body
distractor move  another body moves too
```

### Architecture

```text
current relational scene + I/YOU role query
                  |
                  v
             transformer
                  |
          current entity tokens
                  |
                  +-------------------+
                  |                   |
                  v                   v
           pronoun head       entity-key matching
                                      ^
                                      |
                           20-float recurrent sidecar
                                      ^
                                      |
                           motor-consequence feedback
```

The transformer does momentary interpretation. The recurrent sidecar carries continuity. Zeroing only the sidecar leaves scene/language perception intact.

### Fresh seeds 3 / 4 / 5

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

All seeds pass all precommitted checks.

### What is earned

Safe conclusion:

> A transformer scene/language encoder plus a generic recurrent sidecar can learn a hidden causal-body binding that survives surface-identity changes, persists through silence, rebinds after agency transfer, transfers to a new self-relative task, and causally controls that task under state intervention.

This reconnects the transformer line to Gate 2/3.

### What is not earned

- no consciousness claim;
- no claim that a GRU sidecar is special;
- no claim that this beats a memory token, recurrent KV memory, fast weights, or a generic recurrent transformer;
- the actor readout is allowed to use persistent state as an entity selector;
- backprop still trains the update rule across synthetic lives.

## Run now

```bash
pip install -e '.[torch]'
python experiments/gate4b_hidden_actor.py
```

Single-seed smoke run:

```bash
python experiments/gate4b_hidden_actor.py --quick --seeds 3
```

See `docs/GATE4B_HIDDEN_ACTOR.md`.

## Next scientific road

### Gate 4C — remove the entity-selector bias

Gate 4B's recurrent state is generic, but the actor head explicitly compares persistent state against current entity keys.

Next attacker:

> Can an equal-budget generic recurrent transformer / memory-token architecture discover and use the same factorization without a hand-shaped entity-selector readout?

Matched candidates:

```text
A  Gate-4B recurrent sidecar + entity-key readout
B  persistent memory token
C  recurrent KV state
D  generic GRU/SSM sidecar with unconstrained MLP readout
```

If B/C/D match A, the special entity-selector wiring is unnecessary.

### Local/test-time plasticity remains open

The original stronger Gate 4 question is still alive:

> Can the reusable deictic address be acquired/revised by a local or fast-weight update rule while slow model weights remain fixed?

## Current interpretation of backpropagation

```text
slow backprop
    learns how to interpret scenes
    learns how to update/use persistent state

one particular life
    forward action/consequence stream determines
    which entity that persistent state currently refers to
```

Backprop learns **how to form an I-address**. Forward life determines **which entity occupies it now**.
