import numpy as np
from random import shuffle


class KohonenNework:

    weights = []

    def __init__(self, number_of_weights, learning_rate, epochs):
        self.weights = np.random.rand(number_of_weights, 2)
        self.learning_rate = learning_rate
        self.learning_rate_delta = learning_rate / epochs
        self.epochs = epochs
        self.__class__.weights = self.weights
        self.radius = 6
        self.radius_delta = 1

    def start_training(self, input_cases):

        for _ in xrange(self.epochs):

            for input_case in input_cases:
                output_signals = self.integrate_and_fire(input_case)
                winner_index = output_signals.tolist().index(min(output_signals))
                self.update_weights(winner_index, input_case)

            if self.radius != 0:
                self.radius -= self.radius_delta

            self.learning_rate -= self.learning_rate_delta
            shuffle(input_cases)

    def integrate_and_fire(self, input_case):
        output_signals = np.apply_along_axis(self.inverse_euclidean_distance, 1, self.weights, input_case)
        return output_signals

    def inverse_euclidean_distance(self, weight, input_case):
        return abs(np.linalg.norm(input_case - weight))

    def update_weights(self, winner_index, input_case):
        weight_indices = [((winner_index + i) % len(self.weights)) for i in xrange(-self.radius, self.radius + 1)]
        while len(weight_indices):
            current_index = weight_indices.pop()
            self.apply_weight_delta(current_index, input_case)

    def apply_weight_delta(self, current_index, input_case):
        updated_weight = self.calculate_weight_delta(self.weights[current_index], input_case, 1.0)
        self.weights[current_index] = updated_weight.tolist()

    def calculate_weight_delta(self, current_weight, input_case, neighborhood_factor):
        return np.asarray(current_weight) + (self.learning_rate * neighborhood_factor * (np.array(input_case) - np.array(current_weight)))


