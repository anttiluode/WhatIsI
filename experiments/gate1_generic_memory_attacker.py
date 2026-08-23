import json
from whatisi.gate1 import run_gate1

if __name__ == "__main__":
    print(json.dumps(run_gate1(), indent=2, sort_keys=True))
