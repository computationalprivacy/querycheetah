import pickle

import numpy as np
import multiprocessing
import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import src.dataset_sampler
import defense.back_end
import defense.qbs

import src.limited_syntax_fast_qbs.qbs


def get_seed(args_seed, args_repetition):
    np.random.seed(args_seed)
    seeds = np.random.choice(10 ** 8, 100, replace=False)
    seed = seeds[args_repetition]
    return seed


def sample_attributes(args, data):
    if not bool(args.differentiate_continuous_columns):
        data.sample_attributes(num_attributes=args.num_attributes)
    else:
        if args.attack_type == 'mia':
            raise NotImplementedError
        data.sample_attributes_by_type(num_attributes=args.num_attributes,
                                       num_continuous_attributes=args.num_continuous_attributes)


def get_target_user(args, test_split):
    # We target users who are unique given their non-sensitive attributes.
    if args.attack_type == 'mia':
        # The dataset we load for MIA has the sensitive attribute as the last column.
        unique_users = src.utils.get_indexes_unique(test_split, skip_last_col=True)
    else:
        unique_users = src.utils.get_indexes_unique(test_split)
    tar_user = unique_users[args.target_user_index]
    tar_qbs_seeds = np.random.choice(10 ** 8, size=args.num_target_qbses, replace=False)
    target_record = test_split.iloc[tar_user]
    return tar_user, target_record, tar_qbs_seeds


def get_indexes_unique(dataset, skip_last_col=False):
    """
    Returns the indices of users who are unique given their non-sensitive attributes (pseudo-ids).
    Reused from QuerySnout: https://github.com/computationalprivacy/querysnout.
    """
    num_attributes = dataset.shape[1]
    # Unique in pseudo-IDs: exclude the last (sensitive) column.
    _, idxs, counts = np.unique(dataset.iloc[:, :num_attributes - int(skip_last_col)],
                                axis=0, return_index=True, return_counts=True)
    unique_idxs = sorted(list(idxs[counts == 1]))
    return unique_idxs


def get_train_and_target_qbses_and_ys(args, aux_split, seed, tar_qbs_seeds, tar_user, target_record, test_split):
    train_qbses, y_train, train_indexes = _get_train_and_eval_shadow_qbses(target_record, aux_split, tar_qbs_seeds,
                                                                           args, seed)
    target_qbses, y_test, test_indexes = _get_test_shadow_qbses(test_split, tar_user, tar_qbs_seeds, args)
    indexes = [*train_indexes, *test_indexes] if args.attack_type == 'aia' else [None] * (
                len(train_qbses) + len(target_qbses))
    return target_qbses, train_qbses, y_test, y_train, indexes


def _get_train_and_eval_shadow_qbses(target_record, aux_split, tar_qbs_seeds, args, seed):
    train_qbses, y_train = [], []
    np.random.seed(seed)
    dataset_sampler_tmp = _get_dataset_sampler(target_record, aux_split, args.dataset_size, args.attack_type,
                                               args.eval_fraction, args.num_training_qbses)
    train_datasets = []
    train_indexes = []
    for i in range(args.num_training_qbses):
        qbs_type = 'train' if i < args.num_training_qbses * (1 - args.eval_fraction) else 'eval'
        if args.attack_type == 'aia':
            train_dataset, tar_idx = dataset_sampler_tmp.sample_dataset(qbs_type)
            y_train.append(train_dataset.iloc[tar_idx, -1].item())
            train_indexes.extend(tar_idx)
        else:  # mia
            train_dataset, membership_label = dataset_sampler_tmp.sample_dataset(qbs_type)
            y_train.append(membership_label)
        train_datasets.append(train_dataset)

    training_qbs_seeds = _get_training_qbs_seeds(args.num_training_qbses, tar_qbs_seeds[:args.num_target_qbses])

    for i, train_dataset in enumerate(train_datasets):
        train_qbs = _get_qbs(train_dataset, training_qbs_seeds[i], bool(args.use_limited_syntax_fast_qbs),
                             bool(args.use_shadow_table),
                             bool(args.use_isolating_columns),
                             bool(args.all_uids_for_dynamic_noise),
                             bool(args.generic_noise_to_non_comparison_query),
                             args.qbs)
        train_qbses.append(train_qbs)
    return train_qbses, y_train, train_indexes


def _get_training_qbs_seeds(num_seeds, seeds_to_exclude):
    seeds = set()
    # Sampling seeds that are different from the seeds of the test QBS instances.
    num_trials = 0
    seeds_to_exclude = set(seeds_to_exclude)
    while len(seeds) < num_seeds:
        num_trials += 1
        seed = np.random.randint(10 ** 8)
        if seed in seeds_to_exclude or seed in seeds:
            continue
        else:
            seeds.add(seed)
    seeds = list(seeds)
    assert len(seeds) == num_seeds
    return seeds


def _get_qbs(dataset, qbs_seed, use_limited_syntax_fast_qbs, use_shadow_table=False, use_isolating_columns=False,
             all_uids_for_dynamic_noise=True, generic_noise_to_non_comparison_query=False, qbs='diffix'):
    if qbs == 'diffix':
        if not use_limited_syntax_fast_qbs:
            database = defense.back_end.DuckDBPandas(dataset)
            return defense.qbs.Diffix(database, seed=qbs_seed,
                                      use_shadow_table=use_shadow_table,
                                      use_isolating_columns=use_isolating_columns,
                                      all_uids_for_dynamic_noise=all_uids_for_dynamic_noise,
                                      generic_noise_to_non_comparison_query=generic_noise_to_non_comparison_query)
        else:
            return src.limited_syntax_fast_qbs.qbs.Diffix([tuple(x) for x in dataset.to_numpy()], seed=qbs_seed)
    elif qbs == 'example':
        database = defense.back_end.DuckDBPandas(dataset)
        return defense.qbs.ExampleQBS(database, seed=qbs_seed)


def _get_dataset_sampler(target_record, aux_split, dataset_size, attack_type, eval_fraction, num_training_qbses):
    if attack_type == 'aia':
        return src.dataset_sampler.AuxiliaryWithoutReplacementSamplerAIA(target_record, aux_split, dataset_size)
    else:
        total_num_train_shadow_datasets = int(np.rint((1 - eval_fraction) * num_training_qbses))
        total_num_eval_shadow_datasets = num_training_qbses - total_num_train_shadow_datasets
        return src.dataset_sampler.AuxiliaryWithoutReplacementSamplerMIA(target_record, aux_split, dataset_size,
                                                                         total_num_train_shadow_datasets,
                                                                         total_num_eval_shadow_datasets)


def _get_test_shadow_qbses(test_split, tar_user, tar_qbs_seeds, args):
    if args.attack_type == 'aia':
        target_dataset_sampler = src.dataset_sampler.TargetDatasetSampler(test_split, test_split.iloc[tar_user],
                                                                          args.dataset_size)
    else:
        target_dataset_sampler = src.dataset_sampler.TargetDatasetSamplerMIA(test_split, test_split.iloc[tar_user],
                                                                             args.dataset_size, args.num_target_qbses)
    target_qbses, y_test = [], []
    test_indexes = []
    for i in range(args.num_target_qbses):
        if args.attack_type == 'aia':
            tar_dataset, tar_idx = target_dataset_sampler.sample_dataset(seed=tar_user + i)
            y_test.append(tar_dataset.iloc[tar_idx, -1].item())
            test_indexes.append(tar_idx)
        else:
            tar_dataset, membership_label = target_dataset_sampler.sample_dataset(seed=tar_user + i)
            y_test.append(membership_label)

        tar_qbs = _get_qbs(tar_dataset, tar_qbs_seeds[i], bool(args.use_limited_syntax_fast_qbs),
                           bool(args.use_shadow_table), bool(args.use_isolating_columns),
                           bool(args.all_uids_for_dynamic_noise),
                           bool(args.generic_noise_to_non_comparison_query), args.qbs)
        target_qbses.append(tar_qbs)
    return target_qbses, y_test, test_indexes


def evaluate_queries(list_of_queries, qbses, num_procs, query2query_vector, indexes):
    unique_queries = set()
    new_list_of_queries = []
    for query in list_of_queries:
        if query in unique_queries:
            continue
        new_list_of_queries.append(query)
        unique_queries.add(query)

    list_of_queries = new_list_of_queries
    _evaluate_queries(list_of_queries, num_procs, qbses, query2query_vector, indexes)
    print('cached', len(query2query_vector), 'queries')


def _evaluate_queries(list_of_queries, num_procs, qbses, query2query_vector, indexes):
    if num_procs > 1:
        answers_over_qbses = []
        with multiprocessing.Pool(num_procs) as pool:
            max_ = len(qbses)
            with tqdm.tqdm(total=max_) as pbar:
                for answer in pool.imap(get_answer_for_qbs_and_all_queries,
                                        [(i, list_of_queries, qbs, indexes[i]) for i, qbs in enumerate(qbses)]):
                    answers_over_qbses.append(answer)
                    pbar.update()
    else:
        answers_over_qbses = list(map(get_answer_for_qbs_and_all_queries,
                                      [(i, list_of_queries, qbs, indexes[i]) for i, qbs in enumerate(qbses)]))
    assert np.array_equal(np.array([x[0] for x in answers_over_qbses]), np.arange(len(qbses)))
    for i, query in enumerate(list_of_queries):
        answers_for_the_query = []
        for j in range(len(answers_over_qbses)):
            answers_for_the_query.append(answers_over_qbses[j][1][i])
        query2query_vector[query] = np.array(answers_for_the_query)


def get_answer_for_qbs_and_all_queries(p):
    i, queries, qbs, indx = p
    if hasattr(qbs, 'perform_query') and callable(qbs.perform_query):
        return i, [qbs.perform_query(query) for query in queries]
    else:
        return i, qbs.structured_query([indx], queries)


def get_accuracy_for_list_of_queries(list_of_queries, qbses, ys, query2query_vector, num_procs, num_training_qbses,
                                     eval_fraction, num_target_qbses, indexes):
    X_train, X_eval, X_test = get_X_train_eval_and_test(list_of_queries, qbses, query2query_vector, num_procs,
                                                        num_training_qbses, eval_fraction, num_target_qbses, indexes)
    y_train, y_eval, y_test = get_y_train_eval_end_test(ys, num_training_qbses, eval_fraction, num_target_qbses)
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(random_state=0, max_iter=1000).fit(scaler.transform(X_train), y_train)
    return clf.score(scaler.transform(X_train), y_train), clf.score(scaler.transform(X_eval), y_eval), clf.score(
        scaler.transform(X_test), y_test), clf.coef_


def get_X_train_eval_and_test(list_of_queries, qbses, query2query_answers, num_procs, num_training_qbses, eval_fraction,
                              num_target_qbses, indexes):
    train_size, eval_size, test_size = get_train_eval_test_sizes(num_training_qbses, eval_fraction, num_target_qbses)
    X = np.zeros((num_training_qbses + num_target_qbses, len(list_of_queries)))

    unanswered_queries = get_unanswered_queries(list_of_queries, query2query_answers)
    if len(unanswered_queries) > 0:
        _evaluate_queries(unanswered_queries, num_procs, qbses, query2query_answers, indexes)

    for i, query in enumerate(list_of_queries):
        X[:, i] = query2query_answers[query]
    return X[:train_size, :], X[train_size:-test_size, :], X[-test_size:, :]


def get_unanswered_queries(list_of_queries, query2query_answers):
    unanswered_queries = []
    for i, query in enumerate(list_of_queries):
        if query in query2query_answers:
            continue
        unanswered_queries.append(query)
    return unanswered_queries


def get_y_train_eval_end_test(ys, num_training_qbses, eval_fraction, num_target_qbses):
    train_size, eval_size, test_size = get_train_eval_test_sizes(num_training_qbses, eval_fraction, num_target_qbses)
    assert train_size + eval_size + test_size == len(ys), \
        f'There are {len(ys)} ys there, but there should be {train_size + eval_size + test_size}'
    return ys[:train_size], ys[train_size:-test_size], ys[-test_size:]


def get_train_eval_test_sizes(num_training_qbses, eval_fraction, num_target_qbses):
    train_size = round(num_training_qbses * (1 - eval_fraction))
    test_size = num_target_qbses
    eval_size = num_training_qbses + num_target_qbses - train_size - test_size
    return train_size, eval_size, test_size


def get_result_filename_suffix(args):
    dataset_name = _process_dataset_name_for_filename(args.dataset_name)
    suffix = f'dn-{dataset_name}' + \
             f'_dcc-{args.differentiate_continuous_columns}' + \
             f'_ncc-{args.num_continuous_attributes}' + \
             f'_iters-{args.num_iterations}' + \
             f'_utuv-{args.use_target_user_values}' + \
             f'_nprocs-{args.num_procs}' + \
             f'{"_m" if len(args.note) > 0 else ""}-{args.note}' + \
             f'_in-{args.use_operator_in}' + \
             f'_bw-{args.use_operator_between}' + \
             f'_mt-{args.use_mitigations}' + \
             f'_nin-{args.use_neq_multiple_times}' + \
             f'_cq-{args.change_k_queries_at_each_iteration}' + \
             f'_at-{args.attack_type}' + \
             f'_ols-{args.only_limited_syntax}' + \
             f'_lsfq-{args.use_limited_syntax_fast_qbs}' + \
             f'_ph-{args.only_post_hoc}' + \
             f'_tui-{args.target_user_index}' + \
             f'_rep-{args.repetition}'
    return suffix


def _process_dataset_name_for_filename(dataset_name):
    dataset_name = dataset_name.replace("_with_sensitive", "")
    return dataset_name


def get_column_names(aux_split, attack_type):
    column_names = [*aux_split.columns.values]
    if attack_type == 'aia':
        column_names.append('sens')

    return column_names


def get_probabilities_of_each_value_in_the_aux_split(aux_split, column_names):
    aux_value_probabilities = {}
    for column_name in column_names:
        if column_name == 'sens':
            aux_value_probabilities[column_name] = {1: 0.5, 0: 0.5}
        else:
            value_counts = aux_split[column_name].value_counts(normalize=True)
            aux_value_probabilities[column_name] = {value: probability for (value, probability) in
                                                    zip(value_counts.index.values, value_counts.values)}
    return aux_value_probabilities


def switch_syntax(args, i, syntax_dimensions):
    setattr(args, syntax_dimensions[i], 1 - getattr(args, syntax_dimensions[i]))
