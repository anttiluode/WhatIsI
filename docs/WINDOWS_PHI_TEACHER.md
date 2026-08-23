# Windows Phi teacher

If `llama-cpp-python` dies while constructing `Llama(...)` with:

```text
OSError: [WinError -1073741795] Windows Error 0xc000001d
```

Windows is reporting an **illegal CPU instruction** from the native llama.cpp binary. This is below WhatIsI. It can happen when a prebuilt wheel's CPU/backend build does not match the machine even though installation itself succeeds.

Do not spend a Language Life experiment debugging that wheel unless you specifically want to. WhatIsI can use the same local GGUF through Ollama instead.

## Reuse the GGUF already in this repo

From the repository root, with Ollama installed/running:

```bat
ollama create whatisi-phi -f Modelfile.phi3
ollama run whatisi-phi "Return only this JSON array: []"
```

Then run the teacher:

```bat
python3.13 experiments/gate4a_language_life.py --teacher ollama --ollama-model whatisi-phi --steps 5000
```

The GUI uses the same backend:

```bat
python3.13 experiments/run_language_gui.py --teacher ollama --ollama-model whatisi-phi
```

The teacher is queried only to expand the cached outside-language paraphrase bank. The student transformer, persistent state, symbolic world and learning remain inside WhatIsI.

## Exploration versus learned policy

The training life deliberately uses `epsilon_action=0.10` by default. Roughly one turn in ten is therefore forced random exploration even after the language policy has learned the instruction. A wrong sampled action beside near-zero cross-entropy loss is expected in that regime.

For a cleaner no-exploration continuation/evaluation run:

```bat
python3.13 experiments/gate4a_language_life.py --teacher scripted --resume --steps 500 --epsilon-action 0 --sample-temperature 0
```

Do not compare this greedily sampled accuracy directly with the exploratory training accuracy without noting the policy change.
