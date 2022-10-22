from utils.json_parser import parse_json # json package import

config = parse_json("config.json")
print(config.get('architecture'))