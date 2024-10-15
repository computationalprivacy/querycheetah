import numpy as np
import pandas as pd
import duckdb
from abc import abstractmethod


class Backend:
    @abstractmethod
    def execute_query(self, query):
        pass


class BasePandas(Backend):
    def __init__(self, dataset_or_path_to_dataset):
        if isinstance(dataset_or_path_to_dataset, str):
            self.dataset_path = dataset_or_path_to_dataset
            self.dataset = self._load_dataset()
        else:
            self.dataset = dataset_or_path_to_dataset
        self.dataset = self._initialize(self.dataset)
        self.column_names2index = {column_name: index for index, column_name in enumerate(list(self.dataset.columns)) if
                                   column_name != 'id'}

    def _load_dataset(self):
        return pd.read_csv(self.dataset_path, dtype=np.int32)

    def _initialize(self, dataset):
        dataset['id'] = np.arange(1, len(dataset) + 1)
        dataset.columns = [column_name.lower().replace('-', '') for column_name in dataset.columns]
        dataset.set_index(list(dataset.columns))
        return dataset


class DuckDBPandas(BasePandas):
    def __init__(self, dataset_path):
        super().__init__(dataset_path)
        self.name = 'PandasDuckDB'

    def execute_query(self, query):
        data = self.dataset
        return duckdb.query(query).to_df()
