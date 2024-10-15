import numpy as np
import ast
from abc import abstractmethod
import pyhash

import defense.utils
import defense.parsing


class QBS:
    def __init__(self, database, seed):
        self.database = database
        self.seed = seed

    def get_normal_noise(self, seed, mean=0, std_dev=1):
        return defense.utils.Utils.get_normal_noise(seed=seed, mean=mean, std_dev=std_dev)

    @abstractmethod
    def perform_query(self, query):
        pass


class ExampleQBS(QBS):
    def __init__(self, database, seed):
        super().__init__(database, seed)

    def perform_query(self, query):
        result = self.database.execute_query(query)
        ids = np.array(result.id.values, dtype=np.uint32)
        return len(ids)


class Diffix(QBS):
    pass
