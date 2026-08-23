import json
from whatisi.gate2 import run_gate2

if __name__ == "__main__":
    print(json.dumps(run_gate2(), indent=2, sort_keys=True))
