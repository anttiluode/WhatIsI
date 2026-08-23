import json
from whatisi.gate3 import run_gate3

if __name__ == "__main__":
    print(json.dumps(run_gate3(), indent=2, sort_keys=True))
