import json
from json import JSONEncoder
import numpy

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class Saver:
    def __init__(self, path):
        self.path = path
        self.json = json

    ## Save the model to the given path using JSON format
    def save(self, model):
        with open(self.path, 'w') as f:
            self.json.dump(model, f, cls=NumpyArrayEncoder)
        print(f"Model saved to {self.path}")
    
    ## Load the model from the given path
    def load(self):
        with open(self.path, 'r') as f:
            print(f"Model loaded from {self.path}")
            return self.json.load(f)
        