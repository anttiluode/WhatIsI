# WhatIsI — current handoff

Date: 2026-08-23

## Restart from here

Do **not** restart from the idea that `I` is a special transformer layer.

The current earned object is narrower and stronger:

> **A generic persistent state can act as a reusable deictic address for whichever represented entity currently occupies the learner's causal role. A late lexical system can bind self-produced `I` onto that already-existing address — but a generic unfactorized late transformer does not recover that join efficiently from the same small data budget.**

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
Gate 4D  generic unfactorized late binder                HYPOTHESIS FAILS
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

Train Gate 4B first, freeze it, then introduce only 256 late lexical labels.

The factorized late router is offered three existing address sources:

```text
causal address     frozen Gate-4B persistent state -> current entity
visible speaker    ordinary social role
visible addressee  ordinary social role
```

It sees only speech source (`external / self-produced`) plus pronoun (`I / YOU`). For self-produced `I`, no speaker/body identity is supplied.

The learned routing becomes nearly one-hot:

```text
external + I    -> visible speaker
external + YOU  -> visible addressee
self + I        -> causal address
self + YOU      -> visible addressee
```

Fresh final-evaluator seeds 3/4/5:

```text
self-produced I accuracy        0.9977 / 0.9942 / 0.9933
zero-state self I              0.2309 / 0.2289 / 0.2197
external I                     1.000 / 1.000 / 1.000
external I with state zeroed   1.000 / 1.000 / 1.000
YOU                            1.000 / 1.000 / 1.000
YOU with state zeroed          1.000 / 1.000 / 1.000
counterfactual I state swap    0.9751 / 0.9479 / 0.9028
```

Safe conclusion:

> **A lexical first-person system introduced after causal self-address has already formed can bind self-produced `I` onto that pre-existing address. Removing the address selectively destroys self-`I`; changing it changes the linguistic referent.**

## Gate 4D — Generic Binder Attacker: HYPOTHESIS FAILS

Question:

> If we stop handing the learner `causal address / speaker / addressee` as separate routing options, does a generic architecture rediscover the same decomposition because it is useful?

Keep Gate 4B frozen. Replace Gate 4C's explicit router with a two-layer generic transformer that receives only:

```text
4 current frozen agent representations
raw 20-float persistent state as one generic token
fused lexical query: external/self-produced + I/YOU
visible speaker/addressee name IDs when legitimately available
```

Output is an unconstrained 8-way MLP over current-name classes. There is no precomputed causal-address distribution and no entity-pointer readout.

Matched late-data budget: 256 balanced labels, 800 updates.

Fresh development runs through the final-code path, seeds 3/4/5:

```text
                              seed 3    seed 4    seed 5     mean
self-produced I               0.256     0.431     0.318      0.335
zero-memory self I            0.162     0.151     0.151      0.155
external I                    0.995     0.999     0.999      0.998
external I, memory zero       1.000     1.000     1.000      1.000
YOU                           0.999     1.000     1.000      1.000
YOU, memory zero              1.000     1.000     1.000      1.000
counterfactual I state swap   0.155     0.170     0.190      0.171
```

So the generic binder clearly trains — visible social deixis is solved — but it does **not** recover the causal deictic join. Most importantly, swapping the old causal state does not control the linguistic referent.

Safe conclusion:

> **Gate 4C's factorization is a strong low-data inductive bias, not merely a visualization convenience. The tested generic late transformer does not rediscover that compositional join from the same 256 labels.**

Do not generalize this into "generic architectures can never discover self-address." The result is architecture/data-regime specific. Larger data, end-to-end developmental multi-task training, recurrent KV/SSM memory, or weaker structured biases remain live.

## Run now

```bash
pip install -e '.[torch]'
python experiments/gate4b_hidden_actor.py
python experiments/gate4c_late_i_binding.py
python experiments/gate4d_generic_binder.py
```

Single-seed smoke:

```bash
python experiments/gate4d_generic_binder.py --quick --seeds 3
```

See:

```text
docs/GATE4B_HIDDEN_ACTOR.md
docs/GATE4C_LATE_I_BINDING.md
docs/GATE4D_GENERIC_BINDER.md
```

## Next scientific road

### Find the weakest useful bias

Gate 4D says "remove all routing structure" is too harsh in the matched low-data late-binding regime.

Next question:

> **What is the weakest architectural or training bias that makes the causal address compositional enough for generic downstream language to reuse?**

Candidates:

```text
A  raw memory token + generic transformer             Gate 4D: fails at 256 labels
B  generic cross-attention over current entities      weaker than Gate 4C router
C  pre-language multi-task relational reuse pressure
D  recurrent KV / SSM memory trained end-to-end
E  local fast-weight causal address
```

### Natural-language bridge remains live

Gate 4C/4D use symbolic `external/self-produced × I/YOU` roles. A stronger follow-up would reintroduce varied teacher language while preserving hidden-actor attacks.

### Local/test-time plasticity remains open

The original stronger Gate-4 question is still alive:

> Can the reusable deictic address be acquired/revised by a local or fast-weight update rule while slow weights stay frozen?

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

Backprop learns **how to form/use an I-address**. Forward life determines **which entity occupies it now**. Gate 4C shows a structured late learner can cheaply bind language to it; Gate 4D shows a generic late learner does not automatically discover the same join at equal low-data budget.
