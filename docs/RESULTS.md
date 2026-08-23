# First-session results

Date: 2026-08-23

## Development scar before Gate 0

The first synthetic world was confounded: the causally controlled channel was also unusually energetic. Under that version, a dumb same-size activity-memory attacker reached about `0.805` accuracy.

The world was changed **before the fresh Gate-0 range** so that a non-self channel had deliberately higher variance. The registered causal-pointer thresholds were not lowered to rescue the result.

This scar matters because it is exactly the kind of shortcut the repo is meant to expose.

## Gate 0 — fresh `1000..1039`

```text
accuracy_causal                 1.000000
accuracy_instant                0.678565
accuracy_activity               0.253301
silent_accuracy_causal          1.000000
silent_accuracy_instant         0.200000
mse_causal                      0.001823
mse_uniform                     0.741364
mse_causal / mse_uniform        0.002459
median_recovery_active_steps   14.000000
p90_recovery_active_steps      17.000000
```

Verdict: **PASS** as a coherent local online deictic state.

## Gate 1 — fresh `301..303`

```text
                         pointer      generic GRU
in all                    0.875992      0.936144
in silent                 0.996528      0.960208
in post-transfer          0.569444      0.827315

OOD all                   0.843535      0.901091
OOD silent                0.938889      0.915008
OOD post-transfer         0.546817      0.810359
```

Generic GRU:

```text
persistent hidden floats  4
slow parameters            152
```

Verdict: **generic memory sufficient** under this test. The hand-written pointer has not earned status as a special primitive.

## Abandoned transformer-placement pilot

A small development-only transformer pilot compared no self state, a self prefix, and a per-layer modulation bus. The models mostly learned a shortcut / ignored the supplied state. The pilot was not clean enough to interpret and is **not a gate**.

Do not cite it as evidence for or against a special transformer layer. Return to placement only after the latent itself has earned its role.

## Gate 2 — fresh `401..403`

The recurrent core was trained only on arbitrary self-relative value queries, never owner labels.

```text
downstream NMSE                       0.091017
linear probe owner accuracy           0.957566
counterfactual intervention rate      0.974444
distance to original after injection  1.096171
distance to injected owner value      0.064120
```

Per-seed intervention rate:

```text
401  0.97433
402  0.97500
403  0.97400
```

Verdict: **PASS.** The hidden state is not merely correlated with causal owner; changing it counterfactually changes which entity's current value the query head returns.

## Gate 3 — fresh `501..503`

The Gate-2 recurrent core was frozen. A new binary self-relative task got only 256 indirect task labels and no owner labels. Test evaluation used only silent periods.

```text
                         501      502      503      mean
self-trained core       .9978    .9978    .9982    .99793
random recurrent core   .7716    .7496    .7746    .76527
current sensory         .6598    .6628    .6664    .66300
oracle                  1.0000   1.0000   1.0000   1.00000
```

Verdict: **PASS.** The emergent latent transfers as a reusable deictic factor to a new query family.

## Current interpretation

The first session does **not** support:

```text
I = special neuron
I = special transformer layer
I = dominant PCA component
I = consciousness
```

The narrower object that survived is:

```text
persistent recurrent latent
        +
causal action/consequence binding
        +
reusable across self-relative queries
        +
causally intervenable
```

Call that a **deictic self-address** until a stronger claim is earned.
