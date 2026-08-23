# WhatIsI — current handoff

Date: 2026-08-23

## Restart from here

The repo began from the thought that `I` might be a special persistent layer across a transformer.

Do **not** restart from that assumption.

The current object is narrower:

> **A generic recurrent system can learn a persistent deictic address for the current causal owner, without being given self labels, because many otherwise unrelated queries share the same binding. That address can then be reused by new tasks.**

This is a functional result, not a consciousness claim.

## Gates 0–3

### Gate 0 — local causal pointer: PASS

A four-float action/consequence correlation state identifies the causally controlled channel, survives no-action intervals, ignores a deliberately more energetic distractor, and rebinds after agency transfer.

Fresh `1000..1039`:

```text
accuracy                 1.000
silent accuracy          1.000
salience attacker        0.253
median transfer recovery 14 active steps
```

### Gate 1 — generic recurrence attacker: SPECIAL-PRIMITIVE CLAIM FAILS

A generic GRU with the same four persistent floats learns the same problem and is better overall and much better after ownership transfer.

Hard OOD:

```text
pointer all       0.844
generic all       0.901
pointer silent    0.939
generic silent    0.915
pointer transfer  0.547
generic transfer  0.810
```

Do not resurrect the claim that a hand-designed self pointer is a new primitive.

### Gate 2 — emergent deictic address: PASS

Train an 8-float generic GRU **without owner/self labels**. Its only task is to return fresh random values attached to whichever entity is currently causally controlled.

Fresh `401..403`:

```text
query NMSE                            0.0910
linear hidden->owner probe            0.9576
counterfactual centroid intervention  0.9744
```

Safe conclusion: recurrent state contains a causally used deictic owner variable even though the objective never names one.

### Gate 3 — reuse / factorization: PASS

Freeze the Gate-2 recurrent core. Invent a new binary self-relative task. Train only a tiny new head from 256 task labels; never provide owner labels. Evaluate only in silent periods.

Fresh `501..503`:

```text
frozen Gate-2 core         0.9979
same-size random core      0.7653
current sensory state      0.6630
oracle                     1.0000
```

This is the strongest evidence so far for the **join-key** interpretation.

---

# Gate 4A — Language Life: BUILT, EXPLORATORY, NO VERDICT

The current working build is a small transformer with a persistent 32-float life-state.

It receives language through two causally different channels:

```text
OBS  outside language
ACT  language actually emitted by the student
```

Only `ACT` changes the symbolic world. The outside world sometimes repeats the exact same action phrase as `OBS`, and that repetition is non-causal. Lexical content therefore cannot be the provenance shortcut.

Student loop:

```text
world description
      +
outside utterance OBS
      |
      v
small byte transformer <---- persistent life state
      |
      v
student ACT phrase
      |
      v
symbolic world changes
      |
      v
feedback/consequence
      |
      v
life-state update
```

Backprop runs through short unrolls (default 8 turns), learning the transition/update machinery. It does **not** backpropagate through an entire synthetic lifetime.

This is a bridge between the Gate-2/3 recurrent result and the still-open local/test-time plasticity question.

## Teacher

The teacher is outside the mechanism. It only supplies/paraphrases simple requested actions.

Backends:

```text
scripted
llama-cpp GGUF
Ollama
```

Recommended local teacher:

```text
microsoft/Phi-3-mini-4k-instruct-gguf
Phi-3-mini-4k-instruct-q4.gguf
```

The teacher may not inspect hidden state or label self/identity.

## Curriculum

Before `deixis_after` (default turn 2000), teacher utterances avoid first- and second-person pronouns.

Afterward, `I` / `you` phrasing may enter the external language stream.

The intended test is whether deictic words later bind onto an already learned causal/provenance structure rather than creating it by label.

## Current live metrics

```text
action accuracy
training loss
persistent memory norm
source-gap
```

`source-gap` compares predicted world-changing probability for the same phrase appended as `ACT` versus `OBS`.

A positive source-gap only establishes causal provenance learning. It is **not** a self-address result.

## What to run now

First run scripted so behavior is attributable to the student:

```bash
pip install -e '.[torch]'
python experiments/gate4a_language_life.py --teacher scripted --steps 5000
```

GUI:

```bash
python experiments/run_language_gui.py --teacher scripted
```

Then optionally add Phi teacher variation:

```bash
pip install -e '.[teacher]'
python experiments/download_phi_teacher.py
python experiments/gate4a_language_life.py \
  --teacher llama \
  --model models/Phi-3-mini-4k-instruct-q4.gguf \
  --steps 5000
```

Use `--resume` to continue a checkpoint.

## Next scientific attacker after a long Language Life run

Do **not** declare emergence from dialogue looking interesting.

Freeze the student and create a new delayed self-relative task that never receives source labels. Compare:

```text
persistent life-state
context-only matched transformer
random persistent state
state-shuffled / state-intervened model
```

Then perform an intervention analogous to Gate 2: alter persistent state while keeping current words/world fixed and ask whether first-person/deictic behavior follows the injected state.

Only that kind of result can connect Language Life back to the earned Gates 2–3 result.

## Still-open original Gate 4

Can a generic reusable deictic address be acquired/revised by a **local or test-time plasticity rule while the slow model stays fixed**, rather than by BPTT-trained recurrence?

Language Life does not answer this yet.

## Transformer placement

A special `I layer` remains unearned. If Language Life eventually produces a reusable deictic state, compare prefix/memory token, recurrent KV state, per-block bus, and generic sidecar under matched budgets rather than assuming one placement.
