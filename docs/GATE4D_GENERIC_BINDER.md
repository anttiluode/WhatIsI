# Gate 4D — Generic Binder Attacker

Status: **HYPOTHESIS FAILS in the matched 256-label regime.**

Question:

> If we stop handing the late lexical learner separate `causal address / speaker / addressee` routing channels, does a generic architecture rediscover that decomposition because it is useful?

Gate 4C answered a narrower question positively: after the causal self-address already exists, a tiny factorized lexical router can learn that self-produced `I` points to it. But Gate 4C explicitly offered three candidate referent distributions.

Gate 4D removes that help.

## What the generic learner receives

First train Gate 4B exactly as before, then freeze it.

The late learner receives only:

```text
current frozen agent representations
raw 20-float persistent state as ONE generic token
speech source: external / self-produced
pronoun: I / YOU
visible speaker name when legitimately available
visible addressee name when legitimately available
```

For self-produced `I`, speaker and body identity are both unavailable.

There is no precomputed causal-address distribution, no three-way lexical router, and no entity-key/pointer output head.

The late learner is simply:

```text
CLS
4 current agent tokens
1 raw memory token
1 fused lexical/role query token
        |
        v
2-layer generic transformer
        |
        v
unconstrained MLP -> 8 possible current-name classes
```

So if it wants the Gate-4C solution, it has to discover a computation equivalent to:

```text
raw persistent state
        -> which body does this state refer to?
        -> which current agent token is that body?
        -> which current name does that agent have?
        -> use that name as the referent of self-produced I
```

Nothing in the late readout is hard-coded to perform those joins.

## Matched low-data test

Use the same late-label budget that made Gate 4C almost perfect:

```text
256 balanced lexical labels
800 optimizer updates
2 generic transformer layers
```

Fresh development runs through the final-code training/evaluation path, seeds 3 / 4 / 5:

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

The generic learner therefore clearly trains: visible social deixis is essentially solved.

But it does **not** rediscover the causal deictic join.

Self-produced `I` remains far below the precommitted 0.90 threshold, and the decisive state-intervention rate is only ~0.17 rather than >0.80.

## Why the intervention matters

A generic model could raise ordinary self-`I` accuracy by exploiting correlations between memory and current scenes without actually using memory as a clean referential coordinate.

The counterfactual attack prevents that interpretation.

Keep fixed:

```text
visible world
literal SELF-PRODUCED + I query
generic late-binder weights
frozen Gate-4B weights
```

Replace only the raw persistent state with a centroid associated with another body currently visible in the same world.

If the model had rediscovered the Gate-4C decomposition, the unchanged word `I` should follow the injected body.

It does not.

```text
Gate 4C factorized router        ~0.90-0.98 switch
Gate 4D generic binder           ~0.15-0.19 switch
```

## Verdict

The tested hypothesis fails:

> **A small generic late transformer does not automatically recover the useful deictic factorization from the same 256 labels, even though it easily learns external `I` and `YOU`.**

This is evidence that Gate 4C's factorization is not merely cosmetic interpretability. It is a strong inductive bias for the low-data compositional join.

It does **not** prove that generic architectures can never discover the same decomposition. Larger data, end-to-end developmental training, persistent-token/KV/SSM alternatives, or explicit auxiliary pressures may do so.

The next useful question is therefore no longer "can we remove every bias and still force the same result?" but:

> **What is the weakest architectural/training bias that makes the causal address compositional enough for generic downstream language to reuse?**

Candidate attackers:

```text
A  raw memory token + generic transformer        Gate 4D: fails at 256 labels
B  generic cross-attention to current entities   less explicit than Gate 4C
C  memory trained with multi-task relational reuse before language
D  recurrent KV / SSM memory trained end-to-end
E  local fast-weight causal address
```

## Run

Registered matched-budget attack:

```bash
python experiments/gate4d_generic_binder.py
```

Single-seed smoke:

```bash
python experiments/gate4d_generic_binder.py --quick --seeds 3
```

Stress the generic binder with more late data/updates without changing the gate verdict:

```bash
python experiments/gate4d_generic_binder.py --seeds 3 --labels 4096 --updates 1500
```

A stress run is exploratory unless separately registered; do not replace the 256-label failure with a tuned result.
