from plot_elastic_ring import PlotElasticRing
from tsp_data_parser import TSPDataParser
from kohonen_network import KohonenNetwork
import threading


def start_kohonen_network(data_set):
    cities_normalized = TSPDataParser.parse_to_list(data_set, True)
    kn = KohonenNetwork(len(cities_normalized), 0.5, 400, radius_exp_decay=True, learning_exp_decay=True)
    kn.start_training(cities_normalized)


def main():

    data_sets = {1: 'western_sahara', 2: 'djibouti', 3: 'qatar', 4: 'uruguay'}
    current_set = 3

    if __name__ == '__main__':
        thread = threading.Thread(target=start_kohonen_network, args=(data_sets[current_set],), name="KohonenNetworkThread")
        thread.daemon = True
        thread.start()

        cities_normalized = TSPDataParser.parse_to_list(data_sets[current_set], True)
        per = PlotElasticRing(cities_normalized)
        while True:
            per.update_graph(list(KohonenNetwork.weights))

main()
