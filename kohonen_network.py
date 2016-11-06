import numpy as np
from thread_sync import ThreadSync
import time


class KohonenNetwork:

    weights = []
    learning_rate_epoch = [0]
    radius_epoch = [0]
    current_path_cost = float()

    def __init__(self, number_of_weights, learning_rate, epochs, k, cities, learning_rate_scheme='static', radius_scheme='static'):
        self.weights = np.random.rand(number_of_weights, 2)
        self.__class__.weights = np.copy(self.weights)
        self.k = k
        self.cities = cities

        # Learning Iterations
        self.epochs = epochs
        self.current_epoch = int()

        # Learning Rate
        self.initial_learning_rate = learning_rate
        self.current_learning_rate = self.initial_learning_rate
        self.learning_rate_delta = learning_rate / epochs
        self.learning_rate_scheme = learning_rate_scheme

        # Radius
        self.initial_radius = int(round(float(number_of_weights) / 10))
        self.current_radius = self.initial_radius
        self.radius_delta = 1
        self.radius_scheme = radius_scheme

    def init_report(self):
        print '-------------------------------------------------'
        print 'Initial training values'
        print '-------------------------------------------------'
        print 'Radius:\t\t %i' % self.initial_radius
        print 'Learning Rate:\t %f' % self.initial_learning_rate
        print 'Epochs:\t\t %i' % self.epochs

    def start_training(self, input_cases):

        self.init_report()
        cls = self.__class__
        cls.current_path_cost = self.estimate_current_path(input_cases)
        print "Current path cost: %f" % cls.current_path_cost

        ThreadSync.set()
        ThreadSync.clear()
        time.sleep(5)

        for epoch in xrange(1, self.epochs + 1):

            self.current_epoch = epoch

            cls.learning_rate_epoch.append(float(self.current_learning_rate))
            cls.radius_epoch.append(int(self.current_radius))

            for input_case in input_cases:
                output_signals = self.integrate_and_fire(input_case)
                winner_index = output_signals.tolist().index(min(output_signals))
                self.update_weights(winner_index, input_case)

            if (epoch % self.k) == 0:
                cls.current_path_cost = self.estimate_current_path(input_cases)
                cls.weights = np.copy(self.weights)
                print "Current path cost: %f" % cls.current_path_cost
                ThreadSync.set()
                ThreadSync.clear()
                time.sleep(5)

            self.adjust_radius()
            self.adjust_learning_rate()

        ThreadSync.set()

    """ Routines for firing Neural Network """

    def integrate_and_fire(self, input_case):
        output_signals = np.apply_along_axis(self.euclidean_distance, 1, self.weights, input_case)
        return output_signals

    def euclidean_distance(self, weight, input_case):
        return abs(np.linalg.norm(input_case - weight))

    """ Routines for adjusting Neural Network Weights """

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

    """ Routines for adjusting Decay Rates """

    # TODO Add static and linear decay rate
    def adjust_radius(self):
        if self.radius_scheme == 'exp_decay':
            self.current_radius = int(round(self.initial_radius * pow(0.9, self.current_epoch)))
        elif self.radius_scheme == 'lin_decay' and self.current_radius != 0:
            self.current_radius = max(0, self.initial_radius - int(self.current_epoch * ((self.initial_radius * 1.33) / float(self.epochs))))

    # TODO: Add static learning rate
    def adjust_learning_rate(self):
        if self.learning_rate_scheme == 'exp_decay':
            self.current_learning_rate = self.initial_learning_rate * pow(0.95, self.current_epoch)
        elif self.learning_rate_scheme == 'lin_decay':
            self.current_learning_rate = self.initial_learning_rate - (self.current_epoch * (self.initial_learning_rate / float(self.epochs)))

    """ Routine for calculating current TSP path """

    def estimate_current_path(self, input_cases):
        weights_list = self.weights.tolist()
        cities_to_neurons_mapping = self.map_cities_to_neurons(input_cases, weights_list)
        city_path = self.construct_city_path(cities_to_neurons_mapping, weights_list)
        return self.calculate_city_path_cost(city_path, input_cases)

    def map_cities_to_neurons(self, input_cases, weights_list):
        city_neuron_mapping = dict()
        for i in xrange(len(input_cases)):
            neuron_index = weights_list.index(min(weights_list, key=lambda(weight): self.euclidean_distance(np.asarray(weight), input_cases[i])))
            if neuron_index in city_neuron_mapping:
                city_neuron_mapping[neuron_index].append(i)
            else:
                city_neuron_mapping[neuron_index] = [i]
        return city_neuron_mapping

    def construct_city_path(self, cities_to_neurons_mapping, weight_list):
        city_path = []
        for i in xrange(len(weight_list)):
            if i in cities_to_neurons_mapping:
                city_path += cities_to_neurons_mapping[i]
        return city_path

    def calculate_city_path_cost(self, city_path, input_cases):
        path_cost = float()
        for i in xrange(len(input_cases)):
            path_cost += abs(self.euclidean_distance(np.asarray(self.cities[city_path[i-1]]), self.cities[city_path[i]]))
        return path_cost