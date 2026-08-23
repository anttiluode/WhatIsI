# WhatIsI

**A falsification-driven toy lab asking what computational role, if any, deserves the deictic word `I`.**

This repo does **not** test consciousness, qualia, personhood, or sentience. It asks a smaller question:

> If a learning system acts, receives consequences, persists through time, and repeatedly has to answer questions relative to the same causal agent, does it construct a persistent **deictic self-address** — an internal answer to "which represented entity is currently the source of these actions and owner of their consequences?"

The first session already killed the starting idea that `I` should simply be installed as a special transformer layer.

```text
"I is a special layer"
        |
        v
Gate 0  local causal pointer works
        |
        v
Gate 1  generic 4-float GRU works better
        |      -> special primitive claim dies
        v
Gate 2  generic recurrence, no self labels
        -> self-address emerges anyway
        v
Gate 3  freeze core, invent new task
        -> latent transfers as reusable join key
```

The current hypothesis is:

> **A self-address may be an ordinary recurrent latent variable that learning discovers because many otherwise unrelated predictions share the same deictic binding.**

## Gate 0 — local online deictic state

Four candidate sensorimotor channels are present. Only one is causally controlled by the agent's motor command. Surface-like nuisance features are scrambled and one non-self channel is deliberately more energetic than the controlled channel.

A four-float state updates only from local action/consequence contingency:

```text
e_i(t) = action(t) * consequence_i(t)
score_i <- EMA(score_i, e_i)
```

There is no lifetime backpropagation. With no action, state simply persists.

Fresh `1000..1039`:

```text
causal-pointer accuracy                 1.000
instantaneous causal accuracy           0.679
salience-memory accuracy                0.253
silent-window causal accuracy           1.000
silent-window instantaneous accuracy    0.200
median recovery after agency transfer   14 active steps
```

**PASS.** A tiny local rule can maintain a persistent causal/deictic binding.

## Gate 1 — generic-memory attacker

Replace the hand-written pointer with a generic GRU using the **same four persistent hidden floats**.

Fresh `301..303`:

```text
                         pointer      generic GRU
in-distribution all       0.876          0.936
in-distribution silent    0.997          0.960
post-transfer             0.569          0.827

harder OOD all            0.844          0.901
harder OOD silent         0.939          0.915
harder OOD transfer       0.547          0.810
```

**SPECIAL-PRIMITIVE CLAIM FAILS.** Generic recurrence learns the same binding and adapts faster after ownership changes.

Naming a hidden vector `I` does not make it special.

## Gate 2 — emergent self-address without self labels

Now remove the owner label entirely.

An 8-float generic GRU sees only sensorimotor history. Every step also presents four fresh random values, one attached to each candidate entity. The training target is simply:

```text
return the fresh value attached to the currently causally-controlled entity
```

The random values change every step, so memorizing answers is impossible. The reusable hidden fact is which entity currently occupies the causal first-person role.

Fresh `401..403`:

```text
downstream query NMSE                    0.0910
linear hidden-state -> owner probe       0.9576
```

Then the causal intervention: during silent periods, replace the model's hidden state with the mean state associated with a **different** owner while leaving the current query values unchanged.

```text
output follows injected owner            0.9744
mean distance to original owner's value  1.096
mean distance to injected owner's value  0.064
```

**PASS.** The model was never trained on a self label, yet its recurrent state contains a linearly decodable and causally used deictic owner variable.

This still says nothing about phenomenal consciousness.

## Gate 3 — is the latent reusable, or only task-specific?

Freeze the Gate-2 recurrent core.

Invent a new binary self-relative query task. Train only a tiny new head from **256 task labels** and never provide owner labels. Evaluate only in silent periods where current action/consequence evidence cannot identify the owner.

Fresh `501..503`:

```text
frozen Gate-2 self core    0.9979
same-size random core      0.7653
current sensory state      0.6630
oracle                     1.0000
```

**PASS.** The learned recurrent state is reusable by a new task almost perfectly. That is the strongest evidence so far for the "join key" interpretation.

## Working definition

For this repository, `I` provisionally means:

> **A persistent, reusable deictic state that binds ongoing computation to the entity whose actions have the relevant causal consequences.**

This is only the first layer of a possible hierarchy:

```text
self-address
    which entity is the causal first-person referent?

self-model
    what can that entity do, feel, remember, predict, prefer?

self-narrative
    how is that model expressed across language and autobiography?
```

The experiments here currently support only the first object.

## What happened to the backpropagation problem?

It becomes much cleaner once training and living are separated.

```text
slow learning
    backpropagation may train the machinery / update rule

living
    persistent state changes by forward dynamics
    h_I(t) -> h_I(t+1)
```

Gate 0 is the extreme local case: the state update itself is a simple online correlation rule. Gates 1–3 use a GRU whose transition was learned by backpropagation, but **no backpropagation occurs while the test lifetime runs**. The recurrent state simply evolves forward.

So the live question is not "how do we backpropagate an I through a life?"

It is:

> **Can a sufficiently generic, reusable self-address be acquired and revised by a local/test-time update rule while the slow model stays fixed?**

That is Gate 4.

## Where would this go in a transformer?

A special `I layer` is **not earned**.

A transformer implementation needs persistent state across otherwise separate contexts, but there are several already-plausible placements:

```text
A  persistent prefix / memory token
B  recurrent KV memory
C  per-block residual / FiLM bus
D  generic recurrent or fast-weight sidecar
```

A prefix token can already be attended to by later layers, so privileged access to every block must demonstrate an advantage rather than being assumed.

The likely architecture now looks less like "layer 17 is I" and more like:

```text
slow transformer weights θ
           |
           v
      ordinary blocks  <------ persistent life-state h_I(t)
           |                           ^
           v                           |
         action ----------------> consequence
                                       |
                                       +---- online state update
```

## Next gates

**Gate 4 — local fast updater vs BPTT-trained recurrence.** Can a fast-weight / local prediction-error updater acquire the reusable deictic state while the slow model is frozen?

**Gate 5 — transformer placement.** Only after Gate 4: prefix token vs KV memory vs per-block bus vs sidecar under matched budgets.

See `HANDOFF_CURRENT.md` for the exact restart state and `KILL_LEDGER.md` for claims that may not be quietly resurrected.

## Run

```bash
# Gate 0
pip install -e .
python experiments/gate0_deictic_pointer.py

# Gates 1-3
pip install -e '.[torch]'
python experiments/gate1_generic_memory_attacker.py
python experiments/gate2_emergent_self_address.py
python experiments/gate3_reuse_factorization.py

python -m unittest discover -s tests -v
```

## Repo rule

> **Do not call a state `I` because we named it that. Make the system need a deictic variable, let it learn whatever it learns, then probe, transfer, and intervene on the result.**
