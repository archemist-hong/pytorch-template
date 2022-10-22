import json

def parse_json(config_path: str):
    with open(config_path) as f:
        config = json.load(f)
    return config