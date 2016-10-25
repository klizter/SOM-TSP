import matplotlib.pyplot as plt
from random import uniform


class PlotElasticRing:

    @classmethod
    def plot_elastic_ring(cls, cities, neurons):
        cities_x = [float(city[0]) for city in cities]
        cities_y = [float(city[1]) for city in cities]

        #TODO: Graph not exactly as displayed on data set page, why?
        plt.scatter(cities_x, cities_y, marker='o', c='c')

        neurons_x = [uniform(0.0, 1.0) for _ in xrange(len(cities))]
        neurons_y = [uniform(0.0, 1.0) for _ in xrange(len(cities))]

        plt.scatter(neurons_x, neurons_y, marker='v', c='crimson')
        plt.plot(neurons_x, neurons_y, c='crimson')

        plt.show()