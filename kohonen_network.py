import numpy as np


class KohonenNework:

    weights = []

    def __init__(self, number_of_weights, learning_rate, epochs):
        self.weights = np.random.rand(number_of_weights, 2)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.__class__.weights = self.weights
        self.winner_neuron_index = int()

    def start_training(self, input_cases):

        for _ in xrange(self.epochs):

            for input_case in input_cases:
                output_signals = self.integrate_and_fire(input_case)
                self.winner_neuron_index = output_signals.tolist().index(min(output_signals))
                self.apply_weight_delta(input_case)

    def integrate_and_fire(self, input_case):
        output_signals = np.apply_along_axis(self.euclidean_distance, 1, self.weights, input_case)
        return output_signals

    def euclidean_distance(self, weight, input_case):
        return abs(np.linalg.norm(input_case - weight))

    def apply_weight_delta(self, input_case):
        for weight_index in range(len(self.weights)):
            updated_weight = self.weight_delta(self.weights[winner_neuron_index], input_case, neighborhood_function(weight_index))
            self.weights[winner_neuron_index] = updated_weight.tolist()

    def weight_delta(self, weight, input_case, neighborhood_factor):
        return np.array((weight)) + (self.learning_rate * neighborhood_factor * (np.array(input_case) - np.array(weight)))

    def neighboorhood_function(self, current_index):
        distance_from_winner = self.euclidean_distance(self.weights[self.winner_neuron_index], self.weights[current_index])
        if distance_from_winner < 1:
            distance_from_winner = 1
        return 1/(pow(distance_from_winner, 2))

