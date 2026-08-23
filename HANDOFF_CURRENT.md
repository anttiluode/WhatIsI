# WhatIsI — current handoff

Date: 2026-08-23

## Start here

The repo began from the thought that `I` might be a special persistent layer across a transformer.

Do **not** restart from that assumption.

Three gates have already narrowed it.

### Gate 0: coherent local deictic state — PASS

A four-float action/consequence correlation state can identify the causally controlled channel, retain the binding through no-action intervals, ignore a more energetic distractor, and rebind after agency transfer. No backpropagation through the lifetime.

Fresh `1000..1039`:

```text
accuracy                 1.000
silent accuracy          1.000
salience attacker        0.253
median transfer recovery 14 active steps
```

### Gate 1: generic recurrence attacker — SPECIAL-PRIMITIVE CLAIM FAILS

A generic GRU with the same four persistent floats learns the task and beats the hand-written pointer overall and after transfer.

Hard OOD:

```text
pointer all       0.844
generic all       0.901
pointer silent    0.939
generic silent    0.915
pointer transfer  0.547
generic transfer  0.810
```

Safe conclusion: self-address may be an ordinary recurrent latent/inductive factor, not a new primitive.

### Gate 2: emergent self-address — PASS

Train an 8-float generic GRU **without self labels**. Its task is to answer arbitrary fresh random values attached to the currently causally-controlled entity.

Fresh `401..403`:

```text
query NMSE                            0.0910
linear hidden->owner probe            0.9576
counterfactual centroid intervention  0.9744
```

Replacing hidden state with a different owner's centroid makes the model answer near that different entity's current random value.

Safe conclusion:

> A generic recurrent system can invent a causally used deictic owner variable because many arbitrary queries share the same underlying binding.

Do not upgrade this to consciousness.

## Live next question

Gate 3 should test **reuse/factorization**.

Freeze the recurrent core learned in Gate 2. Present new self-relative query families it was never trained to answer. Compare few-shot or zero-shot transfer against:

- raw sensory state;
- generic hidden state from a model trained on a non-self control task;
- explicit owner oracle;
- matched-capacity recurrent baseline retrained end to end.

The thing to earn is:

> the emergent state is a task-independent deictic join key, not merely a task-specific hidden code.

After that, return to the transformer placement question.
