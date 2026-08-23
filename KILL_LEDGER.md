# Kill ledger

Claims placed here may not be quietly resurrected without a new experiment that directly attacks the reason they were killed.

## Killed / demoted

### "I is a special new computational primitive"

**Killed by Gate 1.** A generic GRU with the same four persistent state floats learns the causal-owner task and outperforms the hand-written pointer overall and after ownership transfer.

### "Persistent state is enough to call something self"

**Killed by definition.** Generic memory can persist arbitrary history. WhatIsI requires evidence that state specifically carries a reusable deictic/causal binding.

### "A transformer needs a privileged I layer"

**Not earned.** Gates 4B/4C succeed with a generic recurrent sidecar plus small readouts. Persistent token, KV, SSM, and other equal-budget alternatives remain untested.

### "First-person language demonstrates an internal self"

**Killed as an inference.** Language can model deictic roles without one persistent causal trajectory.

### "Different vocabulary proves self/other provenance"

**Killed as a valid test.** Identical lexical content can arrive through different causal routes.

### "Gate 4A source-gap is a causal self-address"

**Killed.** Gate 4A supplies OBS/ACT source embeddings and directly trains the auxiliary head to distinguish them. Saturated `source-gap` is therefore a provenance-classification result, not evidence that hidden state discovered causal identity.

### "A long Language Life run forces deictic identity"

**Killed by Gate-4A analysis.** Replay can learn phrase->action mappings with zero persistent memory, and most tasks do not require knowing which represented body is causally controlled.

### "Ordinary I/you role accuracy proves linguistic I is bound to the causal self-address"

**Killed by the Gate-4B residual.** Gate 4B originally had two useful but separable mechanisms: a persistent causal-body address and a scene-level `I = speaker`, `YOU = addressee` role task. Mere coexistence did not prove that self-produced `I` used the causal address.

Gate 4C attacks this directly rather than inferring fusion from pronoun accuracy.

### "The self is the largest PCA/eigenvector component"

**Killed conceptually.** Principal components are directions of variance, not automatically persistent identity or causal ownership.

### "Gate 2, 4B, or 4C demonstrates consciousness"

**Prohibited interpretation.** These experiments demonstrate functional, decodable, causally used deictic states in synthetic models. Nothing measures phenomenal experience.

## Earned / resolved

### "Does the Gate-2 state transfer across a new query family?"

**Yes, Gate 3.** A frozen Gate-2 core supports a new self-relative task far better than matched random/current-state baselines.

### "Can a transformer-side system carry a hidden causal-body address?"

**Yes, Gate 4B under the tested architecture.** A tiny transformer handles current relational/language interpretation while a generic 20-float recurrent sidecar carries continuity.

Across fresh seeds 3/4/5:

```text
actor consequence accuracy        ~0.919-0.928
zero-memory actor accuracy        ~0.232-0.254
silent self-body probe            ~0.834-0.944
held-out join NMSE                ~0.068-0.177
zero-memory join NMSE             ~0.927-0.937
counterfactual intervention       ~0.966-0.988
```

The state also survives name/voice swaps and rebinds after control transfer.

### "Can I/you be trained as actual changing roles rather than fixed words?"

**Yes, Gate 4B.** `I` refers to the current speaker and `YOU` to the current addressee; both roles vary every step. Fresh accuracy is about `0.895-0.945`.

### "Can late self-produced I bind to an already-existing causal address?"

**Yes, Gate 4C under the tested factorized lexical router.** Gate 4B is trained first and frozen. A tiny router then receives only speech source (`external` / `self-produced`) and pronoun (`I` / `YOU`) and learns which of three existing address sources to use: causal state, visible speaker, or visible addressee.

Fresh final-evaluator seeds 3/4/5:

```text
self-produced I accuracy            0.998 / 0.994 / 0.993
zero-causal-state self I            0.231 / 0.229 / 0.220
external I with/without state       1.000 / 1.000
YOU with/without state              1.000 / 1.000
counterfactual I state swap         0.975 / 0.948 / 0.903
```

The late router becomes nearly one-hot: self+`I` -> causal address, external+`I` -> speaker, `YOU` -> addressee. A counterfactual causal-state replacement changes the linguistic referent while the lexical situation and visible world remain fixed.

This earns the narrow statement that **linguistic first person can attach late to a pre-existing causal deictic coordinate**.

## Still live

- Can the Gate-4B/4C result survive an **unconstrained readout** that is not explicitly offered entity/address channels?
- Can equal-budget persistent token / recurrent KV / generic SSM or GRU alternatives match the sidecar?
- Can the reusable deictic state be updated by a **local/test-time plasticity rule while slow weights stay frozen**?
- Can natural-language pronouns from a teacher bind onto the same hidden causal coordinate without the factorized Gate-4C lexical router?
- Does explicit factorization improve data/parameter efficiency versus generic recurrence?
- Does a hierarchy emerge: immediate agency pointer -> autobiographical model -> social/narrative self?
