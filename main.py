from plot_elastic_ring import PlotElasticRing
from plot_decay_rates import PlotLearningRate, PlotRadiusRate
from tsp_data_parser import TSPDataParser
from kohonen_network import KohonenNetwork
import threading
from thread_sync import ThreadSync


def start_kohonen_network(data_set, k):
    cities_normalized = TSPDataParser.parse_to_list(data_set, True)
    kn = KohonenNetwork(int(len(cities_normalized) * 1.33), 0.65, 60, k, radius_scheme='exp_decay', learning_rate_scheme='exp_decay')
    kn.start_training(cities_normalized)


def main():

    data_sets = {1: 'western_sahara', 2: 'djibouti', 3: 'qatar', 4: 'uruguay'}
    current_set = 1
    k = 10

    if __name__ == '__main__':
        thread = threading.Thread(target=start_kohonen_network, args=(data_sets[current_set], k,), name="KohonenNetworkThread")
        thread.daemon = True
        thread.start()

        cities_normalized = TSPDataParser.parse_to_list(data_sets[current_set], True)
        per = PlotElasticRing(cities_normalized)
        plr = PlotLearningRate()
        prr = PlotRadiusRate()
        while True:
            per.update_graph(list(KohonenNetwork.weights))
            plr.update_graph(list(KohonenNetwork.learning_rate_epoch))
            prr.update_graph(list(KohonenNetwork.radius_epoch))
            ThreadSync().wait()


main()
