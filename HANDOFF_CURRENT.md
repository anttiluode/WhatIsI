# WhatIsI — current handoff

Date: 2026-08-23

## Restart from here

The repo began from the thought that `I` might be a special persistent layer across a transformer.

Do **not** restart from that assumption.

The current object is narrower and more interesting:

> **A generic recurrent system can learn a persistent deictic address for the current causal owner, without being given self labels, because many otherwise unrelated queries share the same binding. That address can then be reused by new tasks.**

This is a functional result, not a consciousness claim.

## Gate 0 — local causal pointer: PASS

A four-float action/consequence correlation state identifies the causally controlled channel, survives no-action intervals, ignores a deliberately more energetic distractor, and rebinds after agency transfer. No backpropagation occurs through the lifetime.

Fresh `1000..1039`:

```text
accuracy                 1.000
silent accuracy          1.000
salience attacker        0.253
median transfer recovery 14 active steps
```

This establishes that online local deictic learning is coherent.

## Gate 1 — generic recurrence attacker: SPECIAL-PRIMITIVE CLAIM FAILS

A generic GRU with exactly the same four persistent floats learns the same causal-owner problem and is better overall and much better after ownership transfer.

Hard OOD:

```text
pointer all       0.844
generic all       0.901
pointer silent    0.939
generic silent    0.915
pointer transfer  0.547
generic transfer  0.810
```

Safe conclusion: do not call the hand-written pointer a new primitive. A generic recurrent latent can carry the same information.

## Gate 2 — emergent self-address: PASS

Train an 8-float generic GRU **without owner/self labels**. Its only task is to return fresh random values attached to whichever entity is currently causally controlled.

Fresh `401..403`:

```text
query NMSE                            0.0910
linear hidden->owner probe            0.9576
counterfactual centroid intervention  0.9744
```

Replacing hidden state with a different owner's centroid makes the model answer near that different entity's current random value.

Safe conclusion:

> The recurrent state contains a causally used deictic owner variable, even though the training objective never names one.

## Gate 3 — reuse / factorization: PASS

Freeze the Gate-2 recurrent core. Invent a new binary self-relative query task. Train only a tiny new head from 256 task labels; never provide owner labels. Evaluate only during silent periods, where current action/consequence evidence cannot identify the owner.

Fresh `501..503`:

```text
frozen Gate-2 self core    0.9979
same-size random core      0.7653
current sensory state      0.6630
oracle                     1.0000
```

This is the strongest reason so far to call the latent a **join key** rather than a task-specific code: a new task can cheaply reuse it.

## What this says about backpropagation

Do not frame the problem as "how do we backpropagate an I through a lifetime?"

There are at least two timescales:

```text
slow learning
    backprop can train the recurrent/update machinery

living / inference
    persistent state h_I(t) changes by forward dynamics
    h_I(t) -> h_I(t+1)
```

Gate 0 shows a fully local online update can maintain the binding. Gates 1–3 show a learned recurrent transition can construct a more flexible version. The open problem is to combine those virtues: generic, reusable self-state with a biologically/local-style fast update rather than relying on BPTT as the lifetime mechanism.

## Current working definition

For this repo, `I` provisionally means:

> **A persistent, reusable deictic state that binds computation to the entity whose actions have the relevant causal consequences.**

Do not conflate that with a complete self-model, autobiographical memory, narrative identity, or phenomenal consciousness.

## Next road

### Gate 4 — local fast updater versus BPTT-trained recurrence

Build a fast-weight / local prediction-error updater that changes persistent state online while slow feature machinery is frozen. Compare it to the Gate-1/2 GRU under agency transfers, long silent intervals, and novel query reuse.

The thing to earn is not merely accuracy. Ask whether a small local write rule can acquire and revise the same reusable deictic address.

### Gate 5 — transformer placement

Only after Gate 4: compare persistent prefix token, recurrent KV memory, per-block modulation bus, and generic sidecar under matched state budgets.

A special `I layer` is **not** currently earned.
