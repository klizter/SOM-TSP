import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from random import uniform


class PlotElasticRing:

    draw_tick = 0.5  # Seconds
    city_marker = '*'
    city_color = '#f2b32a'
    elastic_ring_node_color = '#30ac4f'
    elastic_ring_node_alpha = 0.75
    elastic_ring_line_color = '#000000'

    def __init__(self, cities):
        plt.figure(1, figsize=(30, 20))
        # Cache class reference
        cls = self.__class__

        # Add cities to scatter
        cities_x = [city[0] for city in cities]
        cities_y = [city[1] for city in cities]
        marker_area = [100 for _ in range(len(cities))]
        self.city_scatter = plt.scatter(cities_x, cities_y, s=marker_area, marker=cls.city_marker, c=cls.city_color)

        # Prepare neuron elastic circle
        self.elastic_ring_plot = plt.plot([], c=cls.elastic_ring_line_color, linewidth=2.0)

        # Prepare neuron scatter
        self.neuron_scatter = plt.scatter([], [], s=marker_area)

        # Set x and y axis limit
        axes = plt.gca()
        axes.set_xlim([-0.1, 1.1])
        axes.set_ylim([-0.1, 1.1])

        # Display graph
        plt.ion()
        plt.show()

    def update_graph(self, neurons):
        if neurons:

            plt.figure(1)

            # Cache class reference
            cls = self.__class__

            plt.pause(cls.draw_tick)

            # Prepare neuron data for plot
            neurons_x = [neuron[0] for neuron in neurons]
            neurons_y = [neuron[1] for neuron in neurons]

            # Complete elastic circle by adding first element to end of list
            first_x, first_y = neurons_x[0], neurons_y[0]
            neurons_x.append(first_x)
            neurons_y.append(first_y)

            # Commit update and draw
            self.elastic_ring_plot[0].set_xdata(neurons_x)
            self.elastic_ring_plot[0].set_ydata(neurons_y)

            self.neuron_scatter.remove()
            marker_area = [100 for _ in range(len(neurons))]
            self.neuron_scatter = plt.scatter(neurons_x, neurons_y, s=marker_area, marker='H',
                                              color=colorConverter.to_rgba(cls.elastic_ring_node_color,
                                                                          alpha=cls.elastic_ring_node_alpha), zorder=3)

            plt.draw()

    @classmethod
    def plot_elastic_ring(cls, cities, neurons):
        cities_x = [float(city[0]) for city in cities]
        cities_y = [float(city[1]) for city in cities]

        #TODO: Graph not exactly as displayed on data set page, why?
        plt.scatter(cities_x, cities_y, marker=cls.city_marker, c=cls.city_color)

        neurons_x = [uniform(0.0, 1.0) for _ in xrange(len(cities))]
        neurons_y = [uniform(0.0, 1.0) for _ in xrange(len(cities))]

        plt.scatter(neurons_x, neurons_y, marker='v', c='crimson')
        plt.plot(neurons_x, neurons_y, c=cls.elastic_ring_node_color)

        plt.show()
