# WhatIsI

> **Status: FROZEN — 2026-08-23**
>
> This repository is being left at Gate 4D. The code, failed hypotheses, passed gates, and unresolved questions are intentionally preserved as they are. No later experiment is implied by this README.

## The question

This began with a vague but concrete thought:

> **What computational thing, if anything, deserves the word `I`?**

Not consciousness. Not qualia. Not personhood. Not a claim about what a human self ultimately is.

The narrower question was whether an acting learning system would benefit from constructing a persistent variable answering something like:

> **Which represented entity is the one whose actions have my consequences?**

And then, later:

> **Can language learn that the word `I` points to that same coordinate?**

The project started with the wrong guess: perhaps `I` should be a special transformer layer or privileged persistent token.

The experiments pushed in almost the opposite direction.

```text
special I primitive?
      |
      v
Gate 0   a tiny causal pointer works
      |
      v
Gate 1   a same-budget generic GRU works better
          -> special primitive claim dies
      |
      v
Gate 2   remove self labels
          -> generic recurrence still forms a causal deictic state
      |
      v
Gate 3   freeze it and invent a new task
          -> the state transfers as a reusable join key
      |
      v
Gate 4A  long language life
          -> useful scaffold, but language/provenance alone does not force self
      |
      v
Gate 4B  hide which visible body the motor stream controls
          -> transformer + recurrent sidecar learns the hidden causal body
      |
      v
Gate 4C  freeze that causal machinery, introduce I later
          -> self-produced I binds to the pre-existing causal address
      |
      v
Gate 4D  remove the neat factorized router
          -> generic late transformer does NOT cheaply rediscover the same join
```

The resulting working picture is:

> **A functional self-address can be an ordinary persistent latent variable that tracks which represented entity currently occupies the learner's causal first-person role. Linguistic `I` can later become a pointer to that variable. But making that coordinate cheaply reusable is not automatic; representation and inductive bias matter.**

---

## Gate ledger

```text
Gate 0   local causal pointer                         PASS
Gate 1   same-size generic-memory attacker            SPECIAL-PRIMITIVE CLAIM FAILS
Gate 2   emergent deictic address, no self labels     PASS
Gate 3   frozen-core reuse on a new task              PASS
Gate 4A  Language Life + optional Phi teacher         USEFUL SCAFFOLD, NO DEICTIC VERDICT
Gate 4B  hidden causal actor + transformer scene      PASS
Gate 4C  late lexical I -> frozen causal address      PASS
Gate 4D  generic unfactorized late binder             HYPOTHESIS FAILS
```

The failures are part of the result.

---

# 1. A persistent pointer is easy. A special `I` primitive is not needed.

## Gate 0 — local causal pointer

A four-float online state tracks which of several channels is causally linked to action. It survives silent periods, ignores a more energetic distractor, and rebinds when agency transfers.

Fresh registered sweep:

```text
causal accuracy                  1.000
silent causal accuracy           1.000
salience/activity attacker       0.253
median transfer recovery         14 active steps
```

So a tiny persistent causal pointer is coherent.

## Gate 1 — generic-memory attacker

A same-size generic GRU is given the same state budget.

Hard OOD:

```text
                         pointer      generic GRU
all                        0.844          0.901
post-transfer              0.547          0.810
silent                     0.939          0.915
```

The generic memory is better overall and much better after ownership transfer.

**Killed claim:** `I` requires a special new computational primitive.

Naming a state `I` earns nothing.

---

# 2. A generic recurrent state can become a deictic address without self labels.

## Gate 2 — emergent self-address

The model never receives an owner/self label.

It receives sensorimotor history plus fresh random values attached to candidate entities and must return the value belonging to the currently causally controlled entity.

Fresh seeds 401–403:

```text
query NMSE                              0.0910
linear hidden-state -> owner probe      0.9576
counterfactual state intervention       0.9744
```

The important result is the intervention. Replace the hidden state with one associated with another owner while holding the current query fixed, and the answer follows the injected owner.

So the state is not merely correlated with causal identity. It is being used as the coordinate that computation is relative to.

## Gate 3 — reuse

Freeze the Gate-2 recurrent core. Invent a different self-relative task. Train only a tiny new head from 256 labels, still without owner labels.

```text
frozen Gate-2 core          0.9979
same-size random core       0.7653
current sensory state       0.6630
oracle                      1.0000
```

This is where the phrase **deictic join key** became justified.

The latent is reusable across tasks that share one hidden question:

> which entity is this computation relative to?

---

# 3. Language by itself did not create the result.

## Gate 4A — Language Life

Gate 4A built a continuing tiny transformer learner with:

```text
outside language
self-emitted language
symbolic actions and consequences
24-float persistent state
replay consolidation
optional local Phi-3 teacher
```

A 5000-turn Phi/Ollama run learned the command language well. Recent action accuracy reached about the expected ceiling under the deliberately retained 10% exploration policy.

But the stronger interpretation died under inspection:

- `OBS` versus `ACT` provenance was explicitly supplied by source embeddings;
- the `source-gap` classifier was directly supervised;
- replay learned many phrase -> action mappings with zero memory;
- most commands did not require tracking a hidden causal entity;
- first-person phrases could often be solved as ordinary lexical patterns.

So Gate 4A remains useful engineering and provenance work, but it is **not evidence of a deictic self-address**.

That failure motivated Gate 4B.

---

# 4. Hide the causal body.

## Gate 4B — Hidden Causal Actor

Each life contains four visible agents.

Each agent has:

```text
stable body marker
mutable name
mutable voice
position
```

Observation order changes every step.

One body is connected to the learner's motor stream, but the model is never told which one.

The world attacks shortcuts:

```text
surface swap      names and voices are reassigned
silent windows    no new motor consequence arrives
control transfer  the motor stream jumps to another body
distractor motion another body also moves
```

The architecture separates current interpretation from temporal continuity:

```text
current relational scene
        |
        v
 tiny transformer
        |
 current entity representations
        |
        +-------------------------+
                                  |
                         persistent 20-float
                         recurrent sidecar
                                  ^
                                  |
                        motor consequences
```

The sidecar is generic recurrence. It is not named `self` during training.

A reproduced full run on seeds 3/4/5 gave:

```text
actor consequence accuracy        0.9247 mean
zero-memory actor accuracy        0.2391 mean
post surface-swap accuracy        0.9922 mean
post transfer accuracy            1.0000 mean
pronoun-role accuracy             0.9113 mean
silent self-body probe            0.8979 mean
held-out join NMSE                0.1208 mean
zero-memory join NMSE             0.9316 mean
counterfactual intervention       0.9729 mean
```

The cleanest ablation is the first pair:

```text
persistent state present    ~0.925
persistent state zeroed     ~0.239
```

With four visible candidates, zeroing the continuing state collapses actor prediction to roughly the 0.25 chance floor while leaving current scene perception intact.

The stronger evidence is again reuse plus intervention:

1. a new 256-label self-relative task reuses the frozen state;
2. zeroing the state destroys that reuse;
3. replacing the state with one corresponding to another body makes the new task answer relative to that counterfactual body.

**What Gate 4B earns:** a transformer-side system can carry a persistent hidden causal-body address.

It does not earn consciousness, a biological mechanism, or a uniquely correct architecture.

---

# 5. Let the word `I` arrive late.

## Gate 4C — Late `I` Binding

Gate 4B still left two things merely coexisting:

```text
causal address      which body is causally mine?
linguistic deixis   I = speaker, YOU = addressee
```

Gate 4C asks whether they can become one functional coordinate.

Developmental order matters:

```text
train Gate 4B
      |
      v
FREEZE ALL CAUSAL MACHINERY
      |
      v
only then introduce late I / YOU labels
```

The late learner sees only:

```text
speech source: external / self-produced
pronoun:       I / YOU
```

It is offered three existing referent sources:

```text
causal address
visible speaker
visible addressee
```

For **self-produced `I`**, no visible speaker/body identity is supplied.

With only 256 late lexical labels, final-code development seeds 3/4/5 produced:

```text
self-produced I accuracy        0.9977 / 0.9942 / 0.9933
zero-state self I              0.2309 / 0.2289 / 0.2197
external I                     1.000 / 1.000 / 1.000
YOU                            1.000 / 1.000 / 1.000
counterfactual I state swap    0.9751 / 0.9479 / 0.9028
```

The late router learns approximately:

```text
external + I    -> visible speaker
external + YOU  -> visible addressee
self + I        -> causal address
self + YOU      -> visible addressee
```

The decisive attack keeps the scene and literal self-produced `I` query unchanged, changes only the old causal state, and asks what `I` refers to.

```text
h(A) + "I"  -> A
h(C) + "I"  -> C
```

So, in this factorized toy system:

> **linguistic first person can attach late to a causal coordinate that existed before the word did.**

That is the strongest positive statement in this repository.

---

# 6. The factorization does not appear for free.

## Gate 4D — Generic Binder Attacker

Gate 4C is deliberately helpful. It hands the late learner three clean candidate referent channels.

Gate 4D removes that help.

The Gate-4B core is frozen exactly as before. The late learner receives only:

```text
4 current frozen agent representations
raw 20-float persistent state as one generic token
external/self-produced + I/YOU lexical information
visible speaker/addressee IDs when legitimately available
```

There is no precomputed causal-address distribution, no three-way router, and no entity-pointer output head.

Instead, a generic two-layer transformer plus unconstrained MLP must discover the entire composition itself.

Matched low-data budget:

```text
256 balanced late labels
800 optimizer updates
2 transformer layers
```

Fresh development seeds 3/4/5:

```text
                              seed 3    seed 4    seed 5     mean
self-produced I               0.256     0.431     0.318      0.335
zero-memory self I            0.162     0.151     0.151      0.155
external I                    0.995     0.999     0.999      0.998
YOU                           0.999     1.000     1.000      1.000
counterfactual I state swap   0.155     0.170     0.190      0.171
```

The generic binder clearly learns. External `I` and `YOU` are essentially solved.

What it does not discover is the compositional chain:

```text
raw persistent state
       -> which body?
       -> which current entity?
       -> which current name?
       -> referent of self-produced I
```

Most importantly, changing the old causal state does not control the linguistic referent.

So the Gate-4D hypothesis fails in the tested regime.

**What this earns:** Gate 4C's factorization is doing real computational work. It is a strong low-data inductive bias, not merely a convenient visualization of something a generic late transformer was already doing.

**What it does not earn:** a universal claim that generic architectures can never discover the same decomposition. The result is specific to this architecture, training history, and data budget.

---

# What we know at the freeze point

The experiments support a narrow functional story.

### 1. `I` does not need to be a special primitive.

A generic persistent state is enough.

### 2. Persistence alone is not enough.

The state becomes interesting only when it carries a reusable relation to the currently causal entity and survives intervention/reuse tests.

### 3. The useful object is deictic.

It behaves less like a rich autobiographical self-model and more like an address or join key:

> **this computation is relative to that entity.**

### 4. The referent is dynamic.

Names, voices, observation slots, and even the controlled body can change. The state can rebind.

### 5. Language can arrive after the causal structure.

A late lexical learner can discover that self-produced `I` points to the pre-existing causal coordinate.

### 6. Cheap reuse depends on structure.

A generic late transformer with substantially more parameters than the tiny factorized router does not recover the same causal linguistic join from the same 256 labels.

That is where this repository stops.

---

# What this repository does **not** show

It does not show:

- consciousness;
- subjective experience;
- personhood;
- that humans implement this exact mechanism;
- that a GRU sidecar is biologically or computationally privileged;
- that `I` is a single scalar/vector in real brains or large language models;
- that generic transformers can never learn the same factorization;
- that the deictic address can yet be acquired using only local/test-time plasticity with slow weights frozen.

The final object here is intentionally modest:

> **a persistent, reusable, causally intervenable deictic coordinate.**

That is a self-address, not a complete self.

```text
self-address
    which represented entity is the causal first-person referent?

self-model
    what can that entity do, remember, predict, value, fear, prefer?

self-narrative
    how is that model expressed through language and autobiography?
```

This repository reaches only the first layer.

---

# Reproduce the frozen gates

Install:

```bash
pip install -e '.[torch]'
```

Core progression:

```bash
python experiments/gate0_deictic_pointer.py
python experiments/gate1_generic_memory_attacker.py
python experiments/gate2_emergent_self_address.py
python experiments/gate3_reuse_factorization.py
python experiments/gate4b_hidden_actor.py
python experiments/gate4c_late_i_binding.py
python experiments/gate4d_generic_binder.py
```

Language Life remains preserved as the exploratory branch that motivated the hidden-actor gates:

```bash
python experiments/gate4a_language_life.py --teacher scripted --steps 5000
```

Tests:

```bash
python -m unittest discover -s tests -v
```

Detailed notes:

```text
docs/GATE4A_LANGUAGE_LIFE.md
docs/GATE4B_HIDDEN_ACTOR.md
docs/GATE4C_LATE_I_BINDING.md
docs/GATE4D_GENERIC_BINDER.md
KILL_LEDGER.md
```

---

# Frozen conclusion

The original question was roughly:

> Could an artificial system have something computationally meaningful that deserves the word `I`?

Within these toys, the answer is neither mystical nor trivial.

There is no evidence here for a special `I` neuron, token, layer, or primitive.

What repeatedly survives is a more ordinary object:

```text
a continuing latent state
        +
action/consequence history
        +
a world containing several possible entities
        |
        v
a reusable coordinate saying
"the current computation is relative to this one"
```

When that coordinate is removed, self-relative computation collapses. When it is transplanted, the referent moves. When a new task arrives, the coordinate can be reused. When the word `I` arrives later, a suitably structured learner can bind the word to it.

But the final attacker matters just as much as the positive gates: a generic late transformer does not cheaply rediscover that join merely because all the raw ingredients are present.

So the frozen conclusion is:

> **A computational `I` may be less like a special object and more like a persistent deictic address — a reusable causal coordinate. The hard part is not storing such a coordinate. The hard part is making the rest of cognition naturally factor through it.**

That is where `WhatIsI` is left.