import json
from whatisi.gate0 import run_gate0

if __name__ == "__main__":
    print(json.dumps(run_gate0(), indent=2, sort_keys=True))
