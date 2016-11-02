import matplotlib.pyplot as plt


class PlotLearningRate:

    learning_rate_color = '#13B016'

    def __init__(self):
        cls = self.__class__
        figure = plt.figure(2)

        first_plot = figure.add_subplot(211)

        self.learning_rate_graph = first_plot.plot([], c=cls.learning_rate_color, linewidth=2.0)
        plt.ylabel("Learning Rate")
        plt.xlabel("Epoch")

        axes = plt.gca()
        axes.set_xlim([0, 100])
        axes.set_ylim([0.0, 1.0])

    def update_graph(self, learning_rate_epoch):
        plt.figure(2)
        self.learning_rate_graph[0].set_xdata([_ for _ in xrange(len(learning_rate_epoch))])
        self.learning_rate_graph[0].set_ydata(learning_rate_epoch)


class PlotRadiusRate:

    radius_color = '#B01326'

    def __init__(self):
        cls = self.__class__
        figure = plt.figure(2)

        first_plot = figure.add_subplot(212)

        self.radius_graph = first_plot.plot([], c=cls.radius_color, linewidth=2.0)
        plt.ylabel("Radius")
        plt.xlabel("Epoch")

        axes = plt.gca()
        axes.set_xlim([0, 50])
        axes.set_ylim([0, 150])

    def update_graph(self, radius_rate_epoch):
        plt.figure(2)
        self.radius_graph[0].set_xdata([_ for _ in xrange(len(radius_rate_epoch))])
        self.radius_graph[0].set_ydata(radius_rate_epoch)
