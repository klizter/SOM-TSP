import numpy as np
from random import shuffle


class KohonenNetwork:

    weights = []
    learning_rate_epoch = []
    radius_epoch = []

    def __init__(self, number_of_weights, learning_rate, epochs, learning_exp_decay=False, radius_exp_decay=False):
        self.weights = np.random.rand(number_of_weights, 2)
        self.__class__.weights = self.weights

        # Learning Iterations
        self.epochs = epochs
        self.current_epoch = int()

        # Learning Rate
        self.initial_learning_rate = learning_rate
        self.current_learning_rate = self.initial_learning_rate
        self.learning_rate_delta = learning_rate / epochs
        self.learning_exp_decay = learning_exp_decay

        # Radius
        self.initial_radius = 7
        self.current_radius = self.initial_radius
        self.radius_delta = 1
        self.radius_exp_decay = radius_exp_decay

    def start_training(self, input_cases):

        cls = self.__class__

        for epoch in xrange(self.epochs):

            self.current_epoch = epoch

            for input_case in input_cases:
                output_signals = self.integrate_and_fire(input_case)
                winner_index = output_signals.tolist().index(min(output_signals))
                self.update_weights(winner_index, input_case)

            self.adjust_radius()
            self.adjust_learning_rate()
            cls.learning_rate_epoch.append(float(self.current_learning_rate))
            cls.radius_epoch.append(int(self.current_radius))
            shuffle(input_cases)

    def adjust_radius(self):
        if self.radius_exp_decay:
            self.current_radius = int(round(self.initial_radius * pow(0.75, self.current_epoch)))
        else:
            #TODO: Fix linear decay for radius
            self.current_radius -= 1

    def adjust_learning_rate(self):
        if self.learning_exp_decay:
            self.current_learning_rate = self.initial_learning_rate * pow(0.9, self.current_epoch)
        else:
            self.current_learning_rate = self.initial_learning_rate - (self.current_epoch * (self.initial_learning_rate / float(self.epochs)))

    def integrate_and_fire(self, input_case):
        output_signals = np.apply_along_axis(self.euclidean_distance, 1, self.weights, input_case)
        return output_signals

    def euclidean_distance(self, weight, input_case):
        return abs(np.linalg.norm(input_case - weight))

    def update_weights(self, winner_index, input_case):
        weight_indices = [((winner_index + i) % len(self.weights)) for i in xrange(-self.current_radius, self.current_radius + 1)]
        while len(weight_indices):
            current_index = weight_indices.pop()
            self.apply_weight_delta(current_index, input_case)

    def apply_weight_delta(self, current_index, input_case):
        updated_weight = self.calculate_weight_delta(self.weights[current_index], input_case, 1.0)
        self.weights[current_index] = updated_weight.tolist()

    def calculate_weight_delta(self, current_weight, input_case, neighborhood_factor):
        return np.asarray(current_weight) + (self.current_learning_rate * neighborhood_factor * (np.array(input_case) - np.array(current_weight)))


