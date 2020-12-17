import numpy as np
from pprint import pprint


class Lamp:
    def __init__(self, *, lamp_id: int, latitude: float, longitude: float):
        self.lamp_id = lamp_id
        self.latitude = latitude
        self.longitude = longitude


class LampCollection:
    def __init__(self):
        self.lamp_collection = set()

    def add_lamp_id(self, *, lamp_id: int):
        self.lamp_collection.add(lamp_id)

    def collection_to_list(self):
        return self.lamp_collection


class LampGridLookup:
    def __init__(
        self,
        n_x_bins: int,
        min_x: float,
        max_x: float,
        n_y_bins: int,
        min_y: float,
        max_y: float,
    ):
        self.min_x = min_x
        self.n_x_bins = n_x_bins
        self.min_x = min_x
        self.max_x = max_x
        self.n_y_bins = n_y_bins
        self.min_y = min_y
        self.max_y = max_y

        self.delta_x = (max_x - min_x) / float(n_x_bins)
        self.delta_y = (max_y - min_y) / float(n_y_bins)
        self.grid = []
        for y in np.arange(self.min_y, self.max_y, self.delta_y):
            row = []
            for x in np.arange(self.min_x, self.max_x, self.delta_x):
                row.append(LampCollection())
            self.grid.append(row)

        self.x_bin_egdes = np.array(
            [min_x + i * self.delta_x for i in range(n_x_bins + 1)]
        )
        self.y_bin_egdes = np.array(
            [min_y + i * self.delta_y for i in range(n_y_bins + 1)]
        )

    def get_bin_index(self, *, x: float, y: float):
        if x < self.min_x:
            raise Exception(
                f" x value ( {x} ) is smaller than minimum allowed value ( {self.min_x} )"
            )
        if x > self.max_x:
            raise Exception(
                f" x value ( {x} ) is larger than maximum allowed value ( {self.max_x} )"
            )
        if y < self.min_y:
            raise Exception(
                f" y value ( {y} ) is smaller than minimum allowed value ( {self.min_y} )"
            )
        if y > self.max_y:
            raise Exception(
                f" y value ( {y} ) is larger than maximum allowed value ( {self.max_y} )"
            )
        # print(f"x={x}, y={y}")
        if x < 0:
            result = np.where(self.x_bin_egdes[:-1][::-1] <= x)
        else:
            result = np.where(self.x_bin_egdes[1:] >= x)
        # print("x_bin_egdes:")
        # print(self.x_bin_egdes[1:])
        # print("x_bin_egdes[:-1][::-1] :")
        # print(self.x_bin_egdes[:-1][::-1])
        # print(result)
        # print(result[0][0])
        x_ind = result[0][0]
        if y < 0:
            result = np.where(self.y_bin_egdes[:-1][::-1] <= y)
        else:
            result = np.where(self.y_bin_egdes[1:] >= y)
        # print("self.max_y:")
        # print(self.max_y)
        # print("y_bin_egdes:")
        # print(self.y_bin_egdes)
        # print("y_bin_egdes[1:]:")
        # print(self.y_bin_egdes[1:])
        # print(result)
        # print(result[0][0])
        y_ind = result[0][0]
        return (x_ind, y_ind)

    def get_collection_at_x_y(self, *, x: float, y: float):
        x_ind, y_ind = self.get_bin_index(x=x, y=y)
        # print("x_ind, y_ind: ")
        # print(x_ind, y_ind)
        # self.print()
        return self.grid[y_ind][x_ind]

    def print(self):
        pprint(self.grid)
