# Gate 4C — Late `I` Binding

Status: **PASS on fresh development seeds 3, 4, 5.**

Gate 4B produced a persistent causal-body address and separately learned ordinary role-swapping `I` / `YOU` semantics. The unresolved question was whether the lexical first person could actually attach to the already-existing causal address.

Gate 4C makes that question deliberately small and hard to misread.

## Developmental order

The causal core is trained first using Gate 4B exactly as before.

Then it is **frozen**.

Only after freezing do we introduce a tiny lexical router trained from 256 late labels.

```text
Gate 4B life
    motor -> consequence -> persistent causal address
                         |
                         v
                     FREEZE CORE
                         |
                         v
                 introduce I / YOU
```

The recurrent self-address machinery cannot reorganize itself to make the late lexical task easier.

## Three possible address sources

The late lexical learner is offered three already-available referent channels:

```text
0  causal address       frozen Gate-4B persistent state -> current entity
1  visible speaker      ordinary social role
2  visible addressee    ordinary social role
```

The lexical router itself sees only:

```text
speech source:  external / self-produced
pronoun:        I / YOU
```

It learns a soft three-way mixture over those address sources.

No body identity enters the router.

## Crucial missing information

For externally heard `I`, the visible speaker is available.

For `YOU`, the visible addressee is available.

For **self-produced `I`**, no speaker name or body identity is supplied at all.

```text
external:   speaker=A says I   -> A
external:   speaker=A says YOU -> addressee
self:       [speaker hidden] I  -> ???
```

The only systematic route to the correct referent in the last case is the pre-existing causal address.

This is not another SELF label. `self-produced` says only that the utterance came through the learner's output/efference channel. It does not identify which visible body currently occupies that causal role.

## Frozen-state ablation

After late lexical training, zero only the old Gate-4B persistent state.

The current visible scene and lexical router remain intact.

Expected signature:

```text
external I   unchanged
YOU          unchanged
self I       collapses toward 4-agent chance
```

Fresh runs through the **final committed evaluator**:

```text
seed     self-I    zero-state self-I    counterfactual I switch
 3       0.9977         0.2309                 0.9751
 4       0.9942         0.2289                 0.9479
 5       0.9933         0.2197                 0.9028

mean     0.9951         0.2265                 0.9419
```

External `I` and `YOU` remained 1.000 in these runs with or without the causal state.

## What the router learned

The learned routing weights become almost one-hot:

```text
external + I    -> visible speaker
external + YOU  -> visible addressee
self + I        -> causal address
self + YOU      -> visible addressee
```

The mapping is learned from late task labels; it is not hard-coded as an `if pronoun == I` branch.

## Counterfactual attack

The decisive attack keeps all of the following fixed:

```text
visible world
literal lexical situation: SELF-PRODUCED + I
late lexical router weights
slow Gate-4B weights
```

Only the old persistent causal state is replaced by a state centroid associated with another body currently visible in the **same held-out world batch**.

Then ask which current visible name `I` refers to.

Across fresh seeds 3 / 4 / 5, the unchanged word `I` follows the injected counterfactual body at:

```text
0.9751 / 0.9479 / 0.9028
```

So the causal intervention is now linguistic:

```text
same scene
same word I
same lexical router

h_I(A) -> "I" refers to A
h_I(C) -> "I" refers to C
```

## What this earns

Safe conclusion:

> **A lexical first-person system introduced only after a causal self-address has already formed can learn to bind self-produced `I` onto that pre-existing address. Removing the causal state selectively destroys self-`I`, and counterfactual state replacement moves the linguistic referent while the word and world remain fixed.**

This closes the specific residual left by Gate 4B: causal self-address and linguistic deixis can become one functional coordinate.

## What this does not earn

- The late router is deliberately tiny and factorized. This does not show that an unconstrained transformer would invent the same routing structure unaided.
- The router is given three candidate address channels. The experiment asks **which one late language binds to**, not whether the entire address architecture emerges from raw text.
- Gate 4B's recurrent update rule was still learned by backprop across lives.
- Nothing here measures consciousness, phenomenal selfhood, or personhood.

The next attacker should therefore remove the explicit three-channel lexical router and compare equal-budget generic architectures.

## Run

```bash
python experiments/gate4c_late_i_binding.py
```

Smoke run:

```bash
python experiments/gate4c_late_i_binding.py --quick --seeds 3
```

The registered run uses Gate 4B's standard configuration, seeds 3 / 4 / 5, and 256 late lexical labels.
