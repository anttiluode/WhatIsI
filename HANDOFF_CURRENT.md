# WhatIsI — current handoff

Date: 2026-08-23

## Restart from here

Do **not** restart from the idea that `I` is a special transformer layer.

The current earned object is narrower and stronger:

> **A generic persistent state can act as a reusable deictic address for whichever represented entity currently occupies the learner's causal role. A late lexical system can bind self-produced `I` onto that already-existing address.**

Functional result only; not a consciousness claim.

## Gate ledger

```text
Gate 0   local 4-float causal pointer                    PASS
Gate 1   same-size generic GRU attacker                  SPECIAL-PRIMITIVE CLAIM FAILS
Gate 2   generic recurrence, no self labels              PASS
Gate 3   freeze core, new self-relative task             PASS
Gate 4A  long language life                              USEFUL SCAFFOLD, NO DEICTIC VERDICT
Gate 4B  hidden causal actor + transformer scene model   PASS
Gate 4C  late lexical I binding onto frozen Gate 4B      PASS
```

## Gates 0–3 in one sentence

A generic recurrent latent can become a causally used, reusable deictic join key without an explicit self label; a new task can reuse it and state intervention changes which entity computation is relative to.

## Gate 4A — Language Life

Useful engineering scaffold only. The 5000-turn Phi run learned language, but OBS/ACT provenance was explicitly supplied, `source-gap` was supervised, replay learned many phrase->action mappings with zero memory, and first-person phrases did not require tracking a hidden causal body.

Do not use Gate 4A as evidence for deictic identity.

## Gate 4B — Hidden Causal Actor: PASS

Each life has four visible agents with a stable body marker, mutable name/voice, and position. Observation order changes every step. One body is controlled by the learner's motor stream, but no SELF/owner label is supplied.

Attacks:

```text
surface swap      names/voices reassigned, body unchanged
silent windows    no new motor consequence
control transfer  motor stream routed to another body
distractor motion another body moves too
```

Architecture:

```text
current scene / role query
          |
          v
     transformer
          |
 current entity tokens
          |
          +----------------------+
          |                      |
          v                      v
 scene-level roles        entity-key matching
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
I/YOU role accuracy               0.895 / 0.945 / 0.912
silent self-body linear probe     0.944 / 0.944 / 0.834
held-out join NMSE                0.068 / 0.121 / 0.177
zero-memory join NMSE             0.927 / 0.937 / 0.935
counterfactual state intervention 0.988 / 0.975 / 0.966
```

The persistent state therefore carries a hidden causal-body address that survives surface identity changes, persists through silence, rebinds, transfers, and is causally intervenable.

## Gate 4C — Late `I` Binding: PASS

This closes the specific Gate-4B residual: the old causal address and ordinary linguistic deixis were both present, but they were not yet forced to be the same coordinate.

### Developmental order

1. Train Gate 4B normally.
2. Freeze the entire Gate-4B core.
3. Only then introduce a tiny lexical router from 256 labels.

The router is offered three already-existing address sources:

```text
causal address     frozen Gate-4B persistent state -> current entity
visible speaker    ordinary social role
visible addressee  ordinary social role
```

It sees only:

```text
speech source  external / self-produced
pronoun        I / YOU
```

For self-produced `I`, **no speaker name/body identity is supplied**. The only systematic route is the old causal address.

The learned routing becomes nearly one-hot:

```text
external + I    -> visible speaker
external + YOU  -> visible addressee
self + I        -> causal address
self + YOU      -> visible addressee
```

Fresh development seeds 3/4/5:

```text
self-produced I accuracy        0.9988 / 0.9942 / 0.9933
zero-state self I              0.2668 / 0.2289 / 0.2197
external I                     1.000 / 1.000 / 1.000
external I with state zeroed   1.000 / 1.000 / 1.000
YOU                            1.000 / 1.000 / 1.000
YOU with state zeroed          1.000 / 1.000 / 1.000
counterfactual I state swap    0.9699 / 0.9583 / 0.9201
```

The decisive intervention keeps the visible world and lexical situation `SELF-PRODUCED + I` fixed, replaces only the old causal state with a centroid for another currently visible body, and asks what `I` refers to. The word follows the injected body in ~92–97% of cases.

Safe conclusion:

> **A lexical first-person system introduced after causal self-address has already formed can bind self-produced `I` onto that pre-existing address. Removing the address selectively destroys self-`I`; changing it changes the linguistic referent.**

This is still a factorized experiment. It does not show that a completely unconstrained transformer invents the router unaided.

## Run now

```bash
pip install -e '.[torch]'
python experiments/gate4b_hidden_actor.py
python experiments/gate4c_late_i_binding.py
```

Single-seed smoke:

```bash
python experiments/gate4c_late_i_binding.py --quick --seeds 3
```

See:

```text
docs/GATE4B_HIDDEN_ACTOR.md
docs/GATE4C_LATE_I_BINDING.md
```

## Next scientific road

### Gate 4D — remove the factorization bias

Gate 4B explicitly uses persistent state as an entity selector. Gate 4C explicitly offers the late lexical learner three possible address channels.

Next attacker:

> Can an equal-budget generic architecture discover the same causal + linguistic factorization without being handed entity/address routing channels?

Matched candidates should include:

```text
A  current Gate-4B/4C factorized sidecar + router
B  persistent memory token
C  recurrent KV state
D  generic GRU/SSM sidecar with unconstrained MLP readout
```

If B/C/D match A, the special routing structure is unnecessary.

### Natural-language bridge remains live

Gate 4C uses symbolic `external/self-produced × I/YOU` lexical roles. A stronger follow-up would reintroduce varied teacher language while preserving the same hidden-actor attacks and ask whether natural first-person sentences bind to the same causal coordinate.

### Local/test-time plasticity remains open

The original stronger Gate-4 question is still alive:

> Can the reusable deictic address be acquired/revised by a local or fast-weight update rule while slow model weights remain frozen?

## Current interpretation of backpropagation

```text
slow backprop
    learns how to interpret scenes
    learns how to update/use persistent state

one particular life
    forward action/consequence stream determines
    which entity that persistent state currently refers to

late lexical learning
    can attach the word I to that already-existing coordinate
```

Backprop learns **how to form/use an I-address**. Forward life determines **which entity occupies it now**. Late lexical learning can then learn **which word points to it**.
