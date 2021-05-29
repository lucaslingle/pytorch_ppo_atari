import numpy as np


class Dataset:
    def __init__(self, data_map):
        self.data_map = data_map
        self.n = next(iter(data_map.values())).shape[0]
        self._next_idx = 0

    def __shuffle(self):
        perm = np.random.permutation(self.n)

        for key in self.data_map:
            self.data_map[key] = self.data_map[key][perm]

        self._next_idx = 0

    def __next_batch(self, batch_size):
        if self._next_idx >= self.n:
            self.__shuffle()

        cur_idx = self._next_idx
        cur_batch_size = min(batch_size, self.n - self._next_idx)

        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][cur_idx:cur_idx+cur_batch_size]

        self._next_idx += cur_batch_size
        return data_map

    def iterate_once(self, batch_size):
        self.__shuffle()
        while self._next_idx <= self.n - batch_size:
            yield self.__next_batch(batch_size)
        self._next_idx = 0

    def subset(self, num_elements):
        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][:num_elements]
        return Dataset(data_map)
