from math import sqrt

class TSPDataParser:

    def __init__(self):
        pass

    @classmethod
    def parse_to_list(cls, filename, normalize=False):
        data_points_file = open('tsp_data_points/' + filename + '.tsp', 'r')
        data_points_list = list()
        x_list = list()
        y_list = list()

        for line in data_points_file:
            point = [float(value) for value in line.strip().split(' ')[1:3]]

            data_points_list.append(point)
            x_list.append(point[0])
            y_list.append(point[1])

        max_x, min_x, max_y, min_y = max(x_list), min(x_list), max(y_list), min(y_list)
        if normalize:
            cls.normalize_points(data_points_list, max_x, min_x, max_y, min_y)

        return data_points_list

    @classmethod
    # https://docs.tibco.com/pub/spotfire/7.0.0/doc/html/norm/norm_scale_between_0_and_1.htm
    def normalize_points(cls, data_points_list, max_x, min_x, max_y, min_y):
        for i in range(len(data_points_list)):
            data_points_list[i][0] = cls.normalize(data_points_list[i][0], min_x, max_x)
            data_points_list[i][1] = cls.normalize(data_points_list[i][1], min_y, max_y)

    @classmethod
    def normalize(cls, value, value_min, value_max):
        return (value - value_min) / (value_max - value_min)

data_points_list = TSPDataParser.parse_to_list('qatar', True)
