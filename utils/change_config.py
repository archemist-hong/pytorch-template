from json_parser import parse_json # json package import
import os
import json
import sys

if __name__ == '__main__':
    argument = sys.argv
    del argument[0]			# 첫번째 인자는 script.py 즉 실행시킨 파일명이 되기 때문에 지운다

    # config chainging
    print('Chaning config.json ...')
    config = parse_json("config.json")
    config.get('traindataset').get('args')['transform'] = argument[0]
    config.get('slack')['query'] = argument[0]
    base_path = "saved/experiment"
    config.get('training')['experiment_path'] = os.path.join(base_path, argument[0])

    with open("config.json", 'w') as outfile:
        json.dump(config, outfile)
