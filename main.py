from plot_elastic_ring import PlotElasticRing
from tsp_data_parser import TSPDataParser
from ann_dummy import ANNDummy
import threading


def main():

    if __name__ == '__main__':
        thread = threading.Thread(target=ANNDummy.generate_random_weights, name="ANNDummyThread")
        thread.daemon = True
        thread.start()

        cities_normalized = TSPDataParser.parse_to_list('western_sahara', True)
        per = PlotElasticRing(cities_normalized)
        while True:
            per.update_graph(list(ANNDummy.neuron_weights))

main()
