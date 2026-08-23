# Gate 4B — Hidden Causal Actor

Status: **PASS on fresh seeds 3, 4, 5. Reproduced independently on the user's Windows/CUDA machine.**

Gate 4A learned language and explicit ACT/OBS provenance, but it did not force the persistent state to carry a deictic identity. Gate 4B removes that shortcut.

## World

Each synthetic life contains four visible agents. Every agent has:

- a stable body marker (`shape`),
- a mutable name,
- a mutable voice,
- a position.

The observation order is randomly permuted every step.

One body is controlled by the student's motor stream, but the model is never given a `SELF` or owner label. It only observes the sensory consequence of its motor command: which visible body/name/voice carried that consequence.

Mid-life attacks:

1. **surface swap** — names and voices are reassigned while the controlled body stays the same;
2. **silent windows** — no motor consequence is available, so the persistent state must survive;
3. **control transfer** — the motor stream is suddenly routed to a different body;
4. **distractor motion** — another body also moves, so visible motion alone is not a self cue.

The supervised target is not "which body is self?". The main target is a consequence prediction:

> Given the current world and the motor stream, which currently visible **name** will carry the motor consequence?

Because names can swap independently of bodies, a successful system benefits from carrying a body-level deictic binding and joining it to the current scene.

## Transformer + persistent sidecar

The architecture deliberately separates momentary interpretation from temporal continuity:

```text
current agent table + motor + deictic query
                    |
                    v
              tiny transformer
                    |
            current scene tokens
                    |
                    +--------------------+
                    |                    |
                    v                    v
             I / YOU role task     entity-key matching
                                         ^
                                         |
                              persistent 20-float state
                                         ^
                                         |
                              motor consequence feedback
```

The transformer never receives the persistent state as an input token. It encodes the current scene. A GRU sidecar carries the continuing deictic state across moments, and the actor head combines that state with current entity tokens.

This placement is not claimed to be uniquely correct. It is useful because zeroing the sidecar leaves current perception/language intact, giving a clean ablation.

## Deictic language

A separate task teaches real role-swapping `I` / `YOU` semantics.

Every step chooses an arbitrary speaker and addressee. The model sees their current names plus either `I` or `YOU` and must return the referred agent's current position.

Thus:

```text
I   -> current speaker
YOU -> current addressee
```

The referent changes constantly. No entity can be memorized as "I".

Important residual: this role task is currently solved by the scene transformer and does **not** require the persistent causal-body state. Gate 4B therefore establishes causal self-address and linguistic role deixis in one system, but does not yet prove that linguistic `I` binds to the persistent causal address when the learner itself is the speaker.

## Held-out reuse test

After training, freeze the core.

During silent periods, attach fresh random values to all eight possible body markers. Train only a tiny new selector from 256 task labels to return the value belonging to the current causal body.

The selector never receives owner labels. It can only reuse whatever body/deictic information already exists in persistent state.

Then perform the Gate-2-style intervention: replace the current persistent state with the centroid associated with a different body while keeping all current query values fixed. If the new head follows the injected body's value, the persistent state is causally acting as a deictic join key.

## Development results

Recorded local runs, seeds 3 / 4 / 5:

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

All three seeds pass the precommitted checks in `whatisi/gate4b.py`.

## Independent reproduction — 2026-08-23

The full default command was rerun from the committed repository on Windows with Python 3.13 / CUDA. All three seeds again passed all precommitted checks.

```text
mean actor consequence accuracy        0.924696
mean zero-memory actor accuracy        0.239149
mean post surface-swap accuracy        0.992188
mean post transfer accuracy            1.000000
mean I/YOU role accuracy               0.911296
mean silent self-body linear probe     0.897917
mean held-out join NMSE                0.120785
mean zero-memory join NMSE             0.931640
mean counterfactual intervention       0.972917
```

Per seed:

```text
seed 3
actor            0.924805     zero-state actor 0.235677
surface swap     0.994141     transfer         1.000000
I/YOU             0.875732     self probe       0.937500
join NMSE         0.076865     zero-state NMSE  0.927018
intervention      0.984375

seed 4
actor             0.919271     zero-state actor 0.249023
surface swap      0.985352     transfer         1.000000
I/YOU             0.944824     self probe       0.943750
join NMSE         0.128918     zero-state NMSE  0.934955
intervention      0.975000

seed 5
actor             0.930013     zero-state actor 0.232747
surface swap      0.997070     transfer         1.000000
I/YOU             0.913330     self probe       0.812500
join NMSE         0.156572     zero-state NMSE  0.932947
intervention      0.959375
```

The PyTorch `enable_nested_tensor ... norm_first` message printed during the run is a performance-path warning, not a failed check.

## What this earns

Safe conclusion:

> **A small transformer for current relational/language interpretation plus a generic persistent recurrent sidecar can learn a hidden causal-body binding that survives surface identity swaps, rebinds after control transfer, remains decodable through silent intervals, transfers to a new self-relative task, and causally controls that task under state intervention.**

This reconnects the transformer line to Gates 2–3.

The zero-memory actor result is especially useful: the actor task has four currently visible candidates, so chance is approximately 0.25. The reproduced zero-memory mean of 0.239 is effectively chance, while the persistent-state model reaches 0.925.

The held-out join result is also substantial: zeroing memory gives NMSE 0.932, while the learned state gives 0.121, an approximately 87% reduction in normalized error on a task that was not used to train the core.

## What this does not earn

- It does not show consciousness or phenomenal selfhood.
- It does not show that a GRU sidecar is biologically special.
- It does not show that this placement beats persistent tokens, recurrent KV memory, or fast weights.
- The actor readout is allowed to use persistent state as an entity selector; the experiment tests whether that state acquires and carries the correct hidden binding, not whether a completely unconstrained transformer invents the entire mechanism unaided.
- Backpropagation still trains the recurrent update rule across synthetic lives. Local/test-time plasticity remains open.
- The linguistic `I` role and the persistent causal-body address are not yet forced to be the same representation. This is the next direct bridge to test.

## Run

```bash
pip install -e '.[torch]'
python experiments/gate4b_hidden_actor.py
```

A smaller smoke run:

```bash
python experiments/gate4b_hidden_actor.py --quick --seeds 3
```

On a CUDA machine the default runner will use CUDA automatically.
