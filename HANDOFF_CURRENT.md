# WhatIsI — frozen handoff

**Frozen: 2026-08-23**

The experimental lineage is intentionally stopped at Gate 4D. Read `README.md` as the final synthesis and `KILL_LEDGER.md` for claims that were explicitly killed or demoted.

Do not restart from the original idea that `I` should be a special transformer layer.

The earned object is narrower:

> **A generic persistent state can become a reusable deictic address for whichever represented entity currently occupies the learner's causal role. A late lexical system can bind self-produced `I` onto that already-existing address, but a generic unfactorized late transformer does not recover the same join efficiently from the matched 256-label budget.**

Functional result only. No consciousness claim.

## Frozen gate ledger

```text
Gate 0   local 4-float causal pointer                    PASS
Gate 1   same-size generic GRU attacker                  SPECIAL-PRIMITIVE CLAIM FAILS
Gate 2   generic recurrence, no self labels              PASS
Gate 3   frozen core, new self-relative task             PASS
Gate 4A  long language life                              USEFUL SCAFFOLD, NO DEICTIC VERDICT
Gate 4B  hidden causal actor + transformer scene model   PASS
Gate 4C  late lexical I binding onto frozen Gate 4B      PASS
Gate 4D  generic unfactorized late binder                HYPOTHESIS FAILS
```

## Final interpretation

```text
slow learning
    learns how to interpret scenes
    learns how to update/use persistent state

one particular life
    action/consequence history determines
    which represented entity occupies the causal role

persistent state
    acts as a reusable deictic coordinate

late language
    can learn that self-produced I points to that coordinate

but
    generic downstream reuse is not automatically cheap
```

The strongest positive evidence comes from reuse plus intervention: new tasks can use the state, and changing the state changes which entity computation is relative to.

The strongest negative evidence comes from Gate 4D: merely giving a generic late transformer the raw persistent state and current entity representations is not enough to make the causal linguistic join appear cheaply in this low-data regime.

## Unresolved, deliberately left unresolved

These are not queued gates. They are simply the boundary of what this repository did not establish:

- whether weaker factorization biases than Gate 4C are sufficient;
- whether recurrent KV, SSM, persistent-token, or other memory architectures discover the same coordinate;
- whether natural varied first-person language binds to the same address without the symbolic lexical setup;
- whether the address can be acquired/revised by local or fast-weight learning while slow weights remain frozen;
- how, if at all, a deictic self-address grows into a richer self-model or self-narrative.

## Reproduce

```bash
pip install -e '.[torch]'
python experiments/gate0_deictic_pointer.py
python experiments/gate1_generic_memory_attacker.py
python experiments/gate2_emergent_self_address.py
python experiments/gate3_reuse_factorization.py
python experiments/gate4b_hidden_actor.py
python experiments/gate4c_late_i_binding.py
python experiments/gate4d_generic_binder.py
```

Exploratory Language Life is preserved separately:

```bash
python experiments/gate4a_language_life.py --teacher scripted --steps 5000
```

No further gate is implied by this handoff.