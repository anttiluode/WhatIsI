# WhatIsI — current handoff

Date: 2026-08-23

## Restart from here

The repo began from the thought that `I` might be a special persistent transformer layer.

Do **not** restart from that assumption.

Current earned object:

> **A generic recurrent system can learn a persistent deictic address for the current causal owner without self labels, because many otherwise unrelated queries share the same binding. New tasks can reuse that address.**

Functional result only; not a consciousness claim.

## Gates 0–3

```text
Gate 0  local 4-float causal pointer                 PASS
        accuracy 1.000; silent 1.000

Gate 1  same-size generic GRU attacker               SPECIAL-PRIMITIVE CLAIM FAILS
        hard OOD all 0.901 vs pointer 0.844
        transfer     0.810 vs pointer 0.547

Gate 2  8-float generic recurrence, no self labels   PASS
        query NMSE 0.0910
        hidden->owner probe 0.9576
        counterfactual state intervention 0.9744

Gate 3  freeze core, invent new self-relative task   PASS
        frozen core 0.9979
        random core 0.7653
        current sensory state 0.6630
```

The Gate-3 result is the strongest reason to call the latent a **reusable deictic join key**.

---

# Gate 4A — Language Life: BUILT, EXPLORATORY, NO VERDICT

The current build asks whether the same kind of causal/deictic factorization can begin to form from language/action experience.

## Final current architecture

```text
hashed word vocabulary        2048
transformer width               64
blocks                           2
heads                            4
persistent life-state           24 floats
actions                         13
```

Language arrives through separate provenance channels:

```text
OBS  outside language
ACT  language actually emitted by the student
```

The same phrase gets the same word-token IDs in either channel. Only the source/provenance embedding differs.

Only `ACT` executes in the symbolic world. The world deliberately repeats some exact student phrases as non-causal `OBS` events.

The action policy receives only:

```text
current world + current outside utterance + persistent life-state
```

It does not rummage through a full transcript. If older experience matters, persistent state must carry it.

## Learning timescales

The development build settled on a useful split:

```text
FAST / LIVING
persistent h(t) updates every turn by forward dynamics

SLOW / CONSOLIDATION
recent heard-language episodes enter a replay buffer
periodically replay trains transformer weights
```

Default short unroll = 8 turns. Default consolidation = every 64 turns, 64 replay updates.

Replay was necessary because a tiny model seeing each elementary phrase once did not reliably consolidate the language mapping. Do not interpret replay as a deictic result; it is the slow-language-learning mechanism.

The original stronger Gate 4 remains open: can the reusable deictic state itself be updated by a local/test-time plasticity rule while slow weights stay frozen?

## Teacher

Teacher backends:

```text
scripted
llama-cpp GGUF
Ollama
```

Recommended local language teacher:

```text
microsoft/Phi-3-mini-4k-instruct-gguf
Phi-3-mini-4k-instruct-q4.gguf
```

The LLM teacher only expands a cached paraphrase bank. It cannot inspect student hidden state, label `self`, or perform the scientific probe.

## Curriculum

Before turn 2000 by default, outside language avoids first- and second-person pronouns.

Afterward, `I` / `you` phrasing may appear.

The eventual question is whether those words attach to causal structure that was already useful, rather than creating the structure by label.

## Development sanity only

One scripted seed was run for 512 turns to make sure the live experiment actually learns:

```text
turn    recent acc    source-gap
  64      0.172        -0.050
 128      0.080        +0.018
 192      0.180        +0.117
 256      0.330        +0.203
 320      0.500        +0.382
 384      0.760        +0.450
 448      0.880        +0.508
 512      0.930        +0.898
```

This is **not a gate**. It says only that the phrase-learning/replay loop and ACT-vs-OBS provenance task are functioning.

A positive `source-gap` is not a self-address result.

## What to run

Start scripted:

```bash
pip install -e '.[torch]'
python experiments/run_language_gui.py --teacher scripted
```

Headless:

```bash
python experiments/gate4a_language_life.py --teacher scripted --steps 5000
```

Optional Phi teacher:

```bash
pip install -e '.[teacher]'
python experiments/download_phi_teacher.py
python experiments/run_language_gui.py \
  --teacher llama \
  --model models/Phi-3-mini-4k-instruct-q4.gguf
```

Use `--resume` to continue the checkpoint in `runs/language_life.pt`.

## Next scientific attacker after a long run

Do **not** declare emergence because dialogue or hidden-state plots look interesting.

Freeze the student and invent a new delayed self-relative task with no source labels. Compare:

```text
persistent life-state
context-only matched transformer
random persistent state
state-shuffled / state-intervened model
```

Then intervene on the persistent state while current words/world remain fixed.

Only reusable transfer plus causal state intervention reconnects Language Life to the earned Gate-2/3 deictic result.

## Transformer placement

A privileged `I layer` remains unearned. If Language Life eventually produces a reusable deictic state, compare prefix token, recurrent KV state, per-block bus, and generic sidecar under matched state budgets.
