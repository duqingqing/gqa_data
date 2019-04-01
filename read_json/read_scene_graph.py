import json


if __name__ == '__main__':
    with open("../train_sceneGraphs.json",'r') as json_file:
        json_data = json.loads(json_file.read())
        print(json_data["2407890"])