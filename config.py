import json

def read_config():
    with open("C:/Users/Wajih/Desktop/Projects Wajih/Graduation/config.json", "r") as f:
        config = json.load(f)
    
    return config