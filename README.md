# WhatIsI

**A falsification-driven toy lab asking whether a useful computational object corresponding to `I` can emerge without being named or installed as a special symbol.**

This repo is not a consciousness detector. It does not claim qualia, sentience, personhood, or a new neural-network primitive.

The narrower question is:

> If a learning system acts, receives consequences, persists through time, and repeatedly has to answer questions relative to the same causal agent, does it benefit from constructing a persistent **deictic self-address** — an internal answer to "which currently represented entity is the source of these actions and the owner of their consequences?"

The first session already changed the hypothesis substantially.

```text
initial thought
    "I is a special layer"
             |
             v
Gate 0: a locally updated causal pointer works
             |
             v
Gate 1 attacker: generic 4-float GRU works better
             |
             v
special primitive claim dies
             |
             v
Gate 2: train generic recurrence without self labels
        on arbitrary self-relative queries
             |
             v
hidden state develops a decodable and causally
manipulable self-address anyway
```

The current live idea is therefore **not** "put an ego module in a transformer."

It is:

> **A self-address may be a reusable latent variable that generic learning discovers because many otherwise unrelated predictions share the same deictic binding.**

---

## Gate 0 — can a persistent deictic pointer exist without backpropagating through a life?

The toy world contains four candidate sensorimotor channels. Only one is causally controlled by the agent's motor command. Surface labels and appearance-like nuisance features are scrambled. One non-self channel is deliberately *more active* than the controlled channel.

A four-float state updates only from local action-consequence contingency:

```text
motor command a(t)
        |
        v
candidate consequences Δx_i(t)
        |
        v
local evidence e_i = a(t) Δx_i(t)
        |
        v
persistent score_i
        |
        v
belief over current causal owner
```

No gradient is propagated through the lifetime. During intervals with no action, the state simply persists.

Fresh seeds `1000..1039`:

```text
causal-pointer accuracy                 1.000
instantaneous causal accuracy           0.679
salience-memory accuracy                0.253
silent-window causal accuracy           1.000
silent-window instantaneous accuracy    0.200
median recovery after agency transfer   14 active steps
private-value MSE / uniform MSE          0.00246
```

**Gate 0 passes.** This only establishes a coherent object: a persistent deictic binding can be learned online from causal contingency with a tiny local rule.

---

## Gate 1 — is that object a special computational primitive?

Attacker: replace the hand-written causal pointer with a completely generic GRU.

The GRU gets exactly the same four persistent hidden floats. It is trained offline on many synthetic lifetimes but receives no privileged architecture for selfhood.

Fresh seeds `301..303`:

```text
                         pointer      generic GRU
in-distribution all       0.876          0.936
in-distribution silent    0.997          0.960
post-transfer             0.569          0.827

harder OOD all            0.844          0.901
harder OOD silent         0.939          0.915
harder OOD transfer       0.547          0.810
```

The generic GRU has 152 slow parameters and four persistent state floats.

**Gate 1 kills the "special self primitive" claim.** Generic recurrence can learn the same causal binding and adapts faster after ownership changes.

That is important: naming a hidden state `I` does not make it special.

---

## Gate 2 — will a self-address emerge when the model is never taught one?

This is the current strongest experiment.

A generic 8-float GRU sees only sensorimotor history. At every time step it is also handed four fresh random values, one attached to each candidate entity. The training target is simply:

```text
return the random value belonging to the currently causally-controlled entity
```

The random values are regenerated every step, so they cannot be memorized. The only reusable solution is to maintain some representation of *which entity currently occupies the causal first-person role*.

Crucially, **the network is never trained on an owner/self label.**

Fresh seeds `401..403`:

```text
downstream query NMSE                    0.0910
linear probe of hidden state -> owner    0.9576
```

Then comes the causal attacker.

For each owner, compute the mean recurrent hidden state on held-out data. During silent periods, replace the model's actual hidden state with the centroid associated with a *different* owner while leaving the current query values unchanged.

```text
counterfactual intervention follows
injected owner                           0.9744

mean distance to original owner's value 1.096
mean distance to injected owner's value 0.064
```

**Gate 2 passes.** In this toy, generic recurrence develops a hidden variable that is:

1. highly predictive of causal owner;
2. learned without self labels;
3. necessary for an arbitrary family of self-relative queries;
4. causally manipulable: moving the hidden state moves which entity the model answers *as*.

That is still not consciousness. But it is much closer to an operational computational meaning of `I` than a special token or a manually named vector.

---

## Current working definition

For this repo, **I** provisionally means:

> **A persistent, reusable deictic state that binds ongoing computation to the entity whose actions have the relevant causal consequences, and which can be reused across otherwise unrelated self-relative queries.**

This definition is intentionally functional and falsifiable.

It does **not** require that the state be explicit, one-dimensional, human-readable, conscious, or implemented by a special neural layer.

---

## So where would this live in a transformer?

The gates so far argue against prematurely installing an `I layer`.

A transformer needs two things that ordinary feed-forward use does not provide by itself:

```text
slow model weights θ
    learned offline / consolidated slowly

persistent life-state h_I(t)
    updated online from action, consequence, memory and prediction error
    carried across otherwise separate contexts
```

The unresolved architecture question is whether `h_I` needs privileged access to every block or whether an ordinary recurrent/persistent token or memory slot is sufficient.

A prefix token can already be attended to by later transformer layers, so **special placement must earn an empirical advantage**.

The next transformer experiment should compare, under matched state/resource budgets:

```text
A  persistent prefix / memory token
B  persistent KV memory
C  per-block FiLM / residual "self bus"
D  generic recurrent sidecar
```

and ask whether any privileged architecture improves compositional transfer rather than merely making training easier.

---

## The deeper hypothesis after Gate 2

The interesting possibility is no longer that `I` is a special neuron, layer, eigenvector, or mass.

It may be a **join key** created by repeated causal structure:

```text
                     persistent deictic address
                              |
          +-------------------+-------------------+
          |                   |                   |
       action              memory              value
          |                   |                   |
       effect              episode             outcome
          |                   |                   |
          +-------------------+-------------------+
                              |
                         same latent role
```

Many tasks ask different questions, but the same binding answers *which entity those questions are relative to*.

That gives a plausible computational reason for a self-model to appear: not metaphysics, but **factorization and reuse**.

---

## What is killed / not earned

See `KILL_LEDGER.md` for the permanent list. Short version:

- `I` is not earned by first-person language.
- `I` is not earned by a large activation or dominant PCA component.
- persistent state alone is memory, not self.
- a hand-written causal pointer is not a new primitive; generic recurrence can learn it.
- a special transformer-wide `I bus` is not currently earned.
- none of these experiments establish consciousness or subjective experience.

---

## Next gates

### Gate 3 — reuse / factorization

Does the emergent hidden self-address transfer cheaply to **new self-relative query families** never used to train the recurrent core?

If yes, the useful object may be a task-independent deictic latent. If every new query requires relearning the whole system, the "join key" interpretation weakens.

### Gate 4 — local fast learning vs lifetime backprop

Freeze the slow network. Allow only a local online update rule for persistent life-state. Compare against a generic recurrent state learned by BPTT and against test-time-memory baselines.

The central question is exactly the one that motivated the repo:

> Can a system *live and update its self-address* without backpropagating through its lived history?

### Gate 5 — transformer placement

Only after Gates 3–4: prefix token vs KV memory vs per-block bus vs sidecar under matched budgets.

---

## Run

Gate 0 only needs NumPy:

```bash
pip install -e .
python experiments/gate0_deictic_pointer.py
```

Gates 1–2 use PyTorch:

```bash
pip install -e '.[torch]'
python experiments/gate1_generic_memory_attacker.py
python experiments/gate2_emergent_self_address.py
```

Tests:

```bash
python -m unittest discover -s tests -v
```

---

## Repo rule

> **Do not call a state `I` because we named it that. Make the system need a deictic variable, let it learn whatever it learns, then probe and intervene on the result.**
