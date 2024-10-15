import numpy as np
import pandas as pd
from scipy.stats import bernoulli


class DatasetSampler(object):
    """
    The idea and structure of this class and the classes inheriting from it are reused from QuerySnout's repository:
    https://github.com/computationalprivacy/querysnout.
    """

    def sample_dataset(self):
        raise NotImplementedError


class DatasetSamplerAIA(DatasetSampler):
    @staticmethod
    def add_randomized_sensitive_attribute(dataset):
        sensitive_attribute = bernoulli.rvs(p=0.5, size=len(dataset))
        dataset['sens'] = sensitive_attribute
        return dataset


class DatasetSamplerMIA(DatasetSampler):
    @staticmethod
    def get_membership_labels(num_shadow_datasets):
        return bernoulli.rvs(p=0.5, size=num_shadow_datasets)


class AuxiliaryWithoutReplacementSamplerAIA(DatasetSamplerAIA):

    def __init__(self, target_record, auxiliary_dataset, dataset_size):
        assert dataset_size < len(auxiliary_dataset), (f'ERROR: Cannot sample without replacement '
                                                       f'{dataset_size}/{len(auxiliary_dataset)} records.')
        self.dataset_size = dataset_size
        self.target_record = target_record

        # We assume the target record is the unique given their non-sensitive attributes.
        auxiliary_dataset_without_target = [aux_record
                                            for _, aux_record in auxiliary_dataset.iterrows()
                                            if tuple(aux_record) != tuple(self.target_record)]

        # Partition the auxiliary dataset D_{aux} into (1) a partition D^{train}_{aux} for sampling the training shadow
        # datasets D_1^{train}, ..., D_f^{train}, and (2) a partition D^{train}_{val} for sampling the validation shadow
        # datasets D_1^{val}, ..., D_g^{val}
        self.split = dict()
        self.split['train'] = pd.DataFrame(
            auxiliary_dataset_without_target[:len(auxiliary_dataset_without_target) // 2])
        self.split['eval'] = pd.DataFrame(auxiliary_dataset_without_target[len(auxiliary_dataset_without_target) // 2:])

    def sample_dataset(self, eval_type):
        dataset_size = min(len(self.split[eval_type]), self.dataset_size - 1)

        # Sample records without replacement from the corresponding partition
        indexes = np.random.choice(len(self.split[eval_type]), size=dataset_size, replace=False)

        other_records = self.split[eval_type].iloc[indexes]

        # Add the target user's record at a random position
        idx_target = np.random.choice(dataset_size + 1)

        dataset = pd.concat((other_records[:idx_target],
                             self.target_record.to_frame().T,
                             other_records[idx_target:]))

        # We assume the sensitive attribute is randomized
        dataset = DatasetSamplerAIA.add_randomized_sensitive_attribute(dataset)
        return dataset, [idx_target]


class AuxiliaryWithoutReplacementSamplerMIA(DatasetSamplerMIA):
    def __init__(self, target_record, auxiliary_dataset, dataset_size, total_num_train_shadow_datasets,
                 total_num_eval_shadow_datasets):
        assert dataset_size < len(auxiliary_dataset), (f'ERROR: Cannot sample without replacement '
                                                       f'{dataset_size}/{len(auxiliary_dataset)} records.')
        self.dataset_size = dataset_size
        self.target_record = target_record

        self.num_sampled_datasets = {'train': 0, 'eval': 0}
        self.membership_labels = {'train': DatasetSamplerMIA.get_membership_labels(total_num_train_shadow_datasets),
                                  'eval': DatasetSamplerMIA.get_membership_labels(total_num_eval_shadow_datasets)}

        # We assume the target record is the unique given their non-sensitive attributes.
        auxiliary_dataset_without_target = [aux_record
                                            for _, aux_record in auxiliary_dataset.iterrows()
                                            if tuple(aux_record) != tuple(self.target_record)]

        # Partition the auxiliary dataset D_{aux} into (1) a partition D^{train}_{aux} for sampling the training shadow
        # datasets D_1^{train}, ..., D_f^{train}, and (2) a partition D^{train}_{val} for sampling the validation shadow
        # datasets D_1^{val}, ..., D_g^{val}
        self.split = dict()
        self.split['train'] = pd.DataFrame(
            auxiliary_dataset_without_target[:len(auxiliary_dataset_without_target) // 2])
        self.split['eval'] = pd.DataFrame(auxiliary_dataset_without_target[len(auxiliary_dataset_without_target) // 2:])

    def sample_dataset(self, eval_type):
        membership_label = self.membership_labels[eval_type][self.num_sampled_datasets[eval_type]]
        # Add the target user's record to the shadow dataset if and only if the membership label is 1.
        if membership_label == 1:
            dataset_size = min(len(self.split[eval_type]), self.dataset_size - 1)
            indexes = np.random.choice(len(self.split[eval_type]), size=dataset_size, replace=False)

            other_records = self.split[eval_type].iloc[indexes]

            idx_target = np.random.choice(dataset_size + 1)

            dataset = pd.concat((other_records[:idx_target],
                                 self.target_record.to_frame().T,
                                 other_records[idx_target:]))
        else:
            dataset_size = min(len(self.split[eval_type]), self.dataset_size)
            indexes = np.random.choice(len(self.split[eval_type]), size=dataset_size, replace=False)
            other_records = self.split[eval_type].iloc[indexes]
            idx_target = -1
            dataset = other_records.copy()

        self.num_sampled_datasets[eval_type] += 1
        return dataset, membership_label


class TargetDatasetSampler(DatasetSamplerAIA):
    def __init__(self, test_split, target_record, dataset_size):
        self.test_split = test_split
        self.target_record = target_record
        self.dataset_size = dataset_size

        # We assume the target record is the unique given their non-sensitive attributes.
        test_split_without_target = [record for _, record in self.test_split.iterrows()
                                     if tuple(record) != tuple(self.target_record)]

        self.test_split = pd.DataFrame(test_split_without_target)

    def sample_dataset(self, seed):
        np.random.seed(seed)
        dataset_size = min(len(self.test_split), self.dataset_size - 1)
        indexes = np.random.choice(len(self.test_split), size=dataset_size,
                                   replace=False)

        other_records = self.test_split.iloc[indexes]

        # We add the target record at a random position in the dataset.
        idx_target = np.random.choice(dataset_size + 1)

        dataset = pd.concat((other_records[:idx_target],
                             self.target_record.to_frame().T,
                             other_records[idx_target:]))

        # Add a randomized column (the sensitive attribute).
        dataset = DatasetSamplerAIA.add_randomized_sensitive_attribute(dataset)
        return dataset, idx_target


class TargetDatasetSamplerMIA(DatasetSamplerMIA):
    def __init__(self, test_split, target_record, dataset_size, total_num_test_shadow_datasets):
        self.test_split = test_split
        self.target_record = target_record
        self.dataset_size = dataset_size
        self.num_sampled_datasets = 0
        self.membership_labels = DatasetSamplerMIA.get_membership_labels(total_num_test_shadow_datasets)

        # We assume the target record is the unique given their non-sensitive attributes.
        test_split_without_target = [record for _, record in self.test_split.iterrows()
                                     if tuple(record) != tuple(self.target_record)]

        self.test_split = pd.DataFrame(test_split_without_target)

    def sample_dataset(self, seed):
        np.random.seed(seed)
        membership_label = self.membership_labels[self.num_sampled_datasets]
        # Add the target user's record to the shadow dataset if and only if the membership label is 1.
        if membership_label == 1:
            dataset_size = min(len(self.test_split), self.dataset_size - 1)
            indexes = np.random.choice(len(self.test_split), size=dataset_size,
                                       replace=False)

            other_records = self.test_split.iloc[indexes]
            # We add the target record at a random position in the dataset.
            idx_target = np.random.choice(dataset_size + 1)

            dataset = pd.concat((other_records[:idx_target],
                                 self.target_record.to_frame().T,
                                 other_records[idx_target:]))
        else:
            dataset_size = min(len(self.test_split), self.dataset_size)
            indexes = np.random.choice(len(self.test_split), size=dataset_size,
                                       replace=False)

            other_records = self.test_split.iloc[indexes]
            dataset = other_records.copy()
            idx_target = -1

        self.num_sampled_datasets += 1
        return dataset, membership_label
