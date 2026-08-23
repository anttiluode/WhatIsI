# Working thesis

## The question

What computational role, if any, deserves the deictic word `I`?

This repo does not start from consciousness. It starts from a control problem:

A persistent agent is represented inside the same world model as other entities. Actions, sensory consequences, memory, goals and social information all need to be assigned to entities. One entity repeatedly occupies a special causal role: its motor commands predict a particular family of consequences, and those consequences matter for subsequent policy.

Repeatedly rediscovering that binding independently for every subsystem is wasteful.

The minimal hypothesis is therefore:

> A self-address is a factorized latent variable that identifies the current causal owner of first-person action/consequence streams and can be reused by many downstream computations.

## Three distinct objects

Do not conflate:

1. **self-address** — which represented entity currently occupies the deictic/causal role;
2. **self-model** — predictions about that entity's body, capabilities, history, preferences and limits;
3. **self-narrative** — language and autobiographical structure referring to that model.

Gate 2 is evidence only for the first object.

## Why backpropagation is not the central mechanism

Backpropagation can train a system that knows how to use persistent state. It does not follow that every event in a deployed agent must be backpropagated through the slow model.

The architecture we eventually want to test separates:

```text
slow parameters θ
    learn general inference / language / world structure

fast persistent state I(t)
    updated online during a lifetime

slow consolidation
    optional later replay from persistent episodes into θ
```

Gate 0 proves only that a toy deictic state can update online without lifetime backpropagation. Gate 1 proves that a generic learned recurrent updater can do better. The open problem is to obtain the genericity of Gate 1 with an online/local update rule rather than BPTT through each lived episode.

## Why the current Gate-2 result matters

The model in Gate 2 receives no self label. It is trained only to answer arbitrary values attached to the causally-controlled entity. A hidden owner code emerges because the same latent binding solves an indefinitely large family of otherwise unrelated queries.

This suggests a stronger experimental program:

> Do not supervise the self variable. Create tasks for which a reusable deictic factor is the cheapest common cause, then see whether the network invents it.

That is the core of WhatIsI.
