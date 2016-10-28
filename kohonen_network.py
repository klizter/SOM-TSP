import numpy as np


class KohonenNework:

    weights = []

    def __init__(self, number_of_weights, learning_rate, epochs):
        self.weights = np.random.rand(number_of_weights, 2)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.__class__.weights = self.weights

    def start_training(self, input_cases):

        for _ in xrange(self.epochs):

            for input_case in input_cases:
                output_signals = self.integrate_and_fire(input_case)
                winner_neuron_index = output_signals.tolist().index(min(output_signals))
                self.apply_weight_delta(winner_neuron_index, input_case)

    def integrate_and_fire(self, input_case):
        output_signals = np.apply_along_axis(self.inverse_euclidean_distance, 1, self.weights, input_case)
        return output_signals

    def inverse_euclidean_distance(self, weight, input_case):
        return abs(np.linalg.norm(input_case - weight))

    def apply_weight_delta(self, neuron_index, input_case):
        updated_weight = self.weight_delta(self.weights[neuron_index], input_case)
        self.weights[neuron_index] = updated_weight.tolist()

    def weight_delta(self, weight, input_case):
        return np.array((weight)) + (self.learning_rate * (np.array(input_case) - np.array(weight)))

    # Smoothing kernel a.k.a neighborhood function
    def hci(self, winner_index, current_index, learning_rate):
        pass

#
# kn = KohonenNework(5, 0.4)
# kn.start_training([[1.0, 1.0]])
