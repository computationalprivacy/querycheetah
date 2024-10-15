import numpy as np
import os
import pandas as pd
import pickle


class DatasetLoader(object):
    """Generic dataset loader.
    Its idea and structure is reused from QuerySnout's repository: https://github.com/computationalprivacy/querysnout.
    """

    def __init__(self, dataset_path, dataset_name, dataset_filename, attack_type):
        self.path = dataset_path
        self.name = dataset_name
        self.dataset_filename = dataset_filename
        self.all_records = self._get_all_records()
        self.attributes = np.arange(len(self.all_records.columns))
        self.attack_type = attack_type
        self.continuous_columns = self._load_names_of_continuous_columns()
        self._print_success()

    def _get_all_records(self):
        """
        Loads the dataset and returns it as a pandas DataFrame.
        """
        dataset = pd.read_csv(os.path.join(self.path, self.name, f'{self.dataset_filename}.csv'))
        dataset.columns = [column_name.lower().replace('-', '') for column_name in dataset.columns]
        return dataset

    def _load_names_of_continuous_columns(self):
        """
        Loads and returns the list of the names of the continuous columns.
        """
        with open(os.path.join(self.path, self.name, f'final_continuous_columns.pickle'), 'rb') as file:
            continuous_columns = pickle.load(file)
        return continuous_columns

    def _print_success(self):
        print(f'Successfully loaded {self.name} of {len(self.all_records)} ' +
              f'records and {len(self.attributes)} attributes.')

    def sample_attributes(self, num_attributes):
        """
        Sample a subset of attributes.
        """
        num_attributes = self._check_num_attributes_valid(num_attributes)
        if self.attack_type == 'mia':
            # When performing MIA experiments, we load a dataset that contains the sensitive column.
            attributes = np.arange(len(self.all_records.columns) - 1)
        else:
            # When performing AIA experiments, the dataset we load does not contain the sensitive column.
            # We add the sensitive column later randomized.
            attributes = np.arange(len(self.all_records.columns))
        self.attributes, _ = self._sampling_helper(attributes, num_attributes)
        print(f'Sampled {num_attributes} attributes : ', self.attributes)
        return self.attributes

    def _check_num_attributes_valid(self, num_attributes):
        if num_attributes is None:
            return len(self.all_records.columns)
        elif isinstance(num_attributes, int):
            if 0 < num_attributes <= len(self.all_records.columns):
                return num_attributes
            else:
                raise ValueError('Invalid `num_attributes` value passed.')
        raise RuntimeError('Invalid `num_attributes` parameter passed.')

    @staticmethod
    def _sampling_helper(dataset, num_samples):
        """
        Returns a sample of size `num_samples` from `dataset`, where the
        records are sampled without replacement.
        """
        if num_samples is None:
            num_samples = len(dataset)
        if type(num_samples) != int:
            raise TypeError('Invalid type for `num_samples`, should be None or int.')
        if num_samples > len(dataset):
            raise ValueError('Invalid value for `num_samples`, it should be smaller than the dataset size.')

        idxs = np.random.choice(len(dataset), num_samples, replace=False)
        return dataset[idxs], idxs

    def sample_attributes_by_type(self, num_attributes, num_continuous_attributes):
        """Sample a subset of attributes by differentiating continuous from non-continuous columns."""
        num_attributes = self._check_num_attributes_valid(num_attributes)
        continuous_attributes_indices = np.array([i for i, column in enumerate(self.all_records.columns)
                                                  if column in self.continuous_columns])
        non_continuous_attributes_indices = np.array([i for i, column in enumerate(self.all_records.columns)
                                                      if column not in self.continuous_columns])
        continuous_attributes, _ = self._sampling_helper(continuous_attributes_indices, num_continuous_attributes)
        non_continuous_attributes, _ = self._sampling_helper(non_continuous_attributes_indices,
                                                             num_attributes - num_continuous_attributes)
        self.attributes = [*continuous_attributes, *non_continuous_attributes]
        print(f'Sampled {num_attributes} attributes : ', self.attributes)
        return self.attributes

    #
    def split_dataset(self, test_size, aux_size, verbose=False):
        """
        Split into a target and auxiliary split.

        test_size: size of the split from which target datasets will be drawn.
        aux_size: size of the split from which shadow datasets will be drawn.
        """
        test_size = self._check_size_valid(test_size, 'test_size')
        aux_size = self._check_size_valid(aux_size, 'aux_size')
        if test_size is None and aux_size is None:
            raise ValueError('Both `test_size` and `aux_size` cannot be None.')
        elif test_size is None:
            test_size = len(self.all_records) - aux_size
        elif aux_size is None:
            aux_size = len(self.all_records) - test_size
        idxs = np.random.choice(len(self.all_records), test_size + aux_size,
                                replace=False)
        self.test_idxs = idxs[:test_size]
        self.auxiliary_idxs = idxs[test_size:]
        if verbose:
            print('Test indexes: ', self.test_idxs)
            print('Auxiliary indexes: ', self.auxiliary_idxs)
        self.test_split = self.all_records.loc[self.test_idxs]
        self.auxiliary_split = self.all_records.loc[self.auxiliary_idxs]
        return self.test_split, self.test_idxs, self.auxiliary_split, \
            self.auxiliary_idxs

    def _check_size_valid(self, size, param_name):
        if size is None:
            return size
        if isinstance(size, int):
            if 0 < size < len(self.all_records):
                return size
            raise ValueError(f'Invalid `{param_name}` value passed.')
        raise RuntimeError(f'Invalid `{param_name}` parameter passed.')

    def get_test_split(self):
        if self.attack_type == 'mia':
            return self.test_split.iloc[:, [*self.attributes, len(self.all_records.columns) - 1]], self.test_idxs
        else:
            return self.test_split.iloc[:, self.attributes], self.test_idxs

    def get_auxiliary_split(self):
        if self.attack_type == 'mia':
            return self.auxiliary_split.iloc[:, [*self.attributes,
                                                 len(self.all_records.columns) - 1]], self.auxiliary_idxs
        else:
            return self.auxiliary_split.iloc[:, self.attributes], self.auxiliary_idxs
