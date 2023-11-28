""" 
Copyright 2023 AIFUTURE LLC.
Code for MMAC23 challenge from AIFUTURE Lab.
"""
import json
# from refile import smart_open


def save_json_items(json_path, json_items):
    with open(json_path, "w") as f:
        f.write("\n".join([json.dumps(x) for x in json_items]))


def load_json_items(json_path):
    def load_json_items(json_path):
        with open(json_path, "r") as f:
            json_items = [json.loads(s) for s in f.readlines()]
        return json_items