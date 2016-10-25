import time
from random import uniform


class ANNDummy:

    def __init__(self):
        pass

    neuron_weights = []

    @classmethod
    def generate_random_weights(cls):
        while True:
            time.sleep(1)
            cls.neuron_weights = [[uniform(0.0, 1.0), uniform(0.0, 1.0)] for _ in xrange(29)]
