from plot_elastic_ring import PlotElasticRing
from tsp_data_parser import TSPDataParser
from kohonen_network import KohonenNetwork
import threading


def start_kohonen_network():
    cities_normalized = TSPDataParser.parse_to_list('western_sahara', True)
    kn = KohonenNetwork(len(cities_normalized), 0.5, 400, radius_exp_decay=True, learning_exp_decay=True)
    kn.start_training(cities_normalized)


def main():

    if __name__ == '__main__':
        thread = threading.Thread(target=start_kohonen_network, name="KohonenNetworkThread")
        thread.daemon = True
        thread.start()

        cities_normalized = TSPDataParser.parse_to_list('western_sahara', True)
        per = PlotElasticRing(cities_normalized)
        while True:
            per.update_graph(list(KohonenNetwork.weights))

main()
