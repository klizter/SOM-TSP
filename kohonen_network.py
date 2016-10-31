import numpy as np
import sys


class KohonenNework:

    weights = []

    def __init__(self, number_of_weights, learning_rate, epochs, radius):
        self.weights = np.random.rand(number_of_weights, 2)
        self.learning_rate = learning_rate
        self.learning_rate_decay_factor = learning_rate*(2/epochs)
        self.epochs = epochs
        self.__class__.weights = self.weights
        self.winner_neuron_index = int()
        self.current_epoch = int()
        self.radius = radius

    def start_training(self, input_cases):
        
        for epoch in xrange(self.epochs):
            self.update_learning_rate()
            self.current_epoch = epoch
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
        weight_of_winner = self.weights[self.winner_neuron_index]
        current_weight_index = 0
        for current_weight in self.weights:
            neighborhood_factor = self.neighborhood_function(current_weight)
            updated_weight = self.weight_delta(current_weight, input_case, neighborhood_factor)
            self.weights[current_weight_index] = updated_weight.tolist()
            current_weight_index += 1

    def weight_delta(self, weight, input_case, neighborhood_factor):
        return np.array((weight)) + \
            (self.learning_rate * neighborhood_factor * \
                (np.array(input_case) - np.array(weight)))

    def neighborhood_function(self, current_weight):
        distance_from_winner = self.euclidean_distance(self.weights[self.winner_neuron_index], current_weight)
        radius = self.radius-(float(self.current_epoch)*(self.radius-0.09))/self.epochs
        if distance_from_winner < radius:
            return 1-2*distance_from_winner
        else:
            return 0
    
    def update_learning_rate(self):
        self.learning_rate -= self.learning_rate_decay_factor

