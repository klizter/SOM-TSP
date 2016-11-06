from plot_elastic_ring import PlotElasticRing
from plot_decay_rates import PlotLearningRate, PlotRadiusRate
from tsp_data_parser import TSPDataParser
from kohonen_network import KohonenNetwork
import threading
from thread_sync import ThreadSync
import matplotlib.pyplot as plt
import os


def start_kohonen_network(cities_normalized, k, epochs, number_of_neurons, initial_learning_rate, decay_scheme, cities):
    kn = KohonenNetwork(number_of_neurons, initial_learning_rate, epochs, k, cities,
                        radius_scheme=decay_scheme, learning_rate_scheme=decay_scheme)
    kn.start_training(cities_normalized)

def main():

    data_sets = {1: 'western_sahara', 2: 'djibouti', 3: 'qatar', 4: 'uruguay'}
    print("1: western_sahara (x-Small)")
    print("2: djibouti (Small)")
    print("3: qatar (Medium)")
    print("4: uruguay (Large)")
    current_set = int(raw_input("Which map?"))

    k = 5
    epochs = 50
    cities = TSPDataParser.parse_to_list(data_sets[current_set])
    cities_normalized = TSPDataParser.parse_to_list(data_sets[current_set], True)
    number_of_neurons = int(len(cities_normalized) * 2)
    initial_learning_rate = 0.65
    decay_scheme = 'lin_decay'


    if __name__ == '__main__':
        thread = threading.Thread(target=start_kohonen_network, args=(cities_normalized, k, epochs, number_of_neurons,
                                                                      initial_learning_rate, decay_scheme, cities), name="KohonenNetworkThread")
        thread.daemon = True
        thread.start()

        per = PlotElasticRing(cities_normalized)
        plr = PlotLearningRate(epochs)
        prr = PlotRadiusRate(epochs)

        save_plot = True
        graph_number = 1
        while True:
            if not ThreadSync.is_set():
                ThreadSync().wait()

            per.update_graph(list(KohonenNetwork.weights))
            plr.update_graph(list(KohonenNetwork.learning_rate_epoch))
            prr.update_graph(list(KohonenNetwork.radius_epoch))

            # Save each plot to file
            if save_plot:
                img_data = (graph_number, number_of_neurons, KohonenNetwork.current_path_cost)
                path = '/Users/ocselvig/Code/AI Programmering/Assignment 3/images/%s/%s/' % (data_sets[current_set], decay_scheme)
                if not os.path.exists(path):
                    os.makedirs(path)
                plt.savefig(path + '%i SOM NoN:%i CPC:%i' % img_data)
                graph_number += 1


main()
