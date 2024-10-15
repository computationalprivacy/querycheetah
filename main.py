import time
import numpy as np
import os
import pickle

import src.parser
import src.dataset_loader
import src.utils
import src.query_searcher


def main():
    # Parse the arguments provided through the command line.
    args = src.parser.Parser.get_args()
    start_time = time.time()

    # Load the dataset
    data = src.dataset_loader.DatasetLoader(args.dataset_path, args.dataset_name, args.dataset_filename,
                                            args.attack_type)

    seed = src.utils.get_seed(args.default_seed, args.repetition)
    np.random.seed(seed)

    # Sample n attributes of the dataset given the current repetition
    src.utils.sample_attributes(args, data)

    # Partition the dataset to simulate the distributions $\mathcal{D}$ and $\mathcal{D}_{aux}$
    test_split_size = len(data.all_records) // 3
    data.split_dataset(test_size=test_split_size, aux_size=None)
    aux_split, aux_idxs = data.get_auxiliary_split()
    test_split, test_idxs = data.get_test_split()

    # The privacy vulnerabilities are record-specific.
    # Get the target record which will be attacked in this run.
    tar_user, target_record, tar_qbs_seeds = src.utils.get_target_user(args, test_split)

    # Sample shadow datasets and instantiate the QBS software on them
    target_qbses, train_qbses, y_test, y_train, indexes = \
        src.utils.get_train_and_target_qbses_and_ys(args, aux_split, seed, tar_qbs_seeds, tar_user, target_record,
                                                    test_split)
    final_ys = np.array([*y_train, *y_test])
    final_qbses = [*train_qbses, *target_qbses]

    # If you have previously discovered an attack against a QBS that implements the main defenses only, you can test
    # whether the mitigations thwart it by setting the flag only_post_hoc.
    suffix = src.utils.get_result_filename_suffix(args)
    if args.only_post_hoc:
        post_hoc_apply_mitigations(args, final_qbses, final_ys, indexes, suffix)
        return

    column_names = src.utils.get_column_names(aux_split, args.attack_type)
    aux_value_probabilities = src.utils.get_probabilities_of_each_value_in_the_aux_split(aux_split, column_names)

    # Search for attacks within the limited syntax Q_lim
    all_accuracies, best_index, best_solution, query2query_answers_dump, fitness = \
        src.query_searcher.QuerySearcher.search_for_this_syntax(args, aux_value_probabilities, column_names,
                                                                data.continuous_columns, final_qbses, final_ys, seed,
                                                                target_record, [], {},
                                                                indexes)
    # Search for attacks within the extended syntax Q_ext
    used_syntax_dimensions = None
    if not args.only_limited_syntax:
        all_accuracies, best_index, best_solution, query2query_answers_dump, used_syntax_dimensions = \
            search_for_attacks_in_extended_syntax(all_accuracies, args, aux_value_probabilities, best_index,
                                                  best_solution, column_names, data, final_qbses, final_ys, fitness,
                                                  indexes, query2query_answers_dump, seed, target_record)

    end_time = time.time()

    # Save the results to the filesystem in the folder specified as a command-line argument `output_dir`
    # (default `results`)
    save_results(all_accuracies, args, best_index, best_solution, end_time, query2query_answers_dump, start_time,
                 suffix, target_record, used_syntax_dimensions)


def post_hoc_apply_mitigations(args, final_qbses, final_ys, indexes, suffix):
    assert args.use_mitigations == 1

    # Load the attack previously discovered against a QBS that does not implement mitigations
    input_suffix = suffix.replace('_mt-1', '_mt-0').replace('_ph-1_', '_ph-0_')
    with open(f'{args.output_dir}/qc_{input_suffix}', 'rb') as f:
        solution = pickle.load(f)['best_solution']

    # Evaluate the queries on the QBS that does implement mitigations
    query2query_answers = {}
    src.utils.evaluate_queries(solution, final_qbses, args.num_procs, query2query_answers, indexes)

    # Evaluate the accuracy of that attack against the QBS with mitigations
    acc = src.utils.get_accuracy_for_list_of_queries(solution, final_qbses, final_ys, query2query_answers,
                                                     args.num_procs,
                                                     args.num_training_qbses, args.eval_fraction,
                                                     args.num_target_qbses, indexes)

    # Save the accuracy to a file
    with open(f'{args.output_dir}/qc_{suffix}', 'wb') as f:
        pickle.dump({'accuracy': acc}, f)


def search_for_attacks_in_extended_syntax(all_accuracies, args, aux_value_probabilities, best_index, best_solution,
                                          column_names, data, final_qbses, final_ys, fitness, indexes,
                                          query2query_answers_dump, seed, target_record):
    used_syntax_dimensions = []
    round2best = {0: (all_accuracies, best_index, best_solution, query2query_answers_dump, fitness, [])}

    # QueryCheetah uses a multi-stage search
    for stage in range(1, len(args.syntax_dimensions) + 1):
        round_results = []
        indices = []

        # In each stage, the syntax is extended along the best extension axis
        # Evaluate the unexplored extension axes one-by-one
        for i in range(len(args.syntax_dimensions)):
            if args.syntax_dimensions[i] in used_syntax_dimensions:
                continue

            # The extension axes are not independent. Check if the syntax can be extended along this axes.
            invalid_settings = False
            src.utils.switch_syntax(args, i, args.syntax_dimensions)
            if args.use_target_user_values == 1 and args.use_operator_between == 1:
                invalid_settings = True

            # Evaluate this unexplored axis
            if not invalid_settings:
                tmp_all_accuracies, tmp_best_index, tmp_best_solution, tmp_query2query_answers_dump, tmp_fitness = \
                    src.query_searcher.QuerySearcher.search_for_this_syntax(args, aux_value_probabilities,
                                                                            column_names, data.continuous_columns,
                                                                            final_qbses, final_ys, seed,
                                                                            target_record, round2best[stage - 1][2],
                                                                            round2best[stage - 1][3], indexes)
            src.utils.switch_syntax(args, i, args.syntax_dimensions)

            # Save the results
            if not invalid_settings:
                round_results.append((tmp_all_accuracies, tmp_best_index, tmp_best_solution,
                                      tmp_query2query_answers_dump, tmp_fitness,
                                      [*used_syntax_dimensions, args.syntax_dimensions[i]]))
                indices.append(i)

        # From the saved results, get the best extension axis for this round
        best_index = np.argmax([round_result[-2] for round_result in round_results])
        round2best[stage] = round_results[best_index]
        used_syntax_dimensions.append(args.syntax_dimensions[indices[best_index]])
        src.utils.switch_syntax(args, indices[best_index], args.syntax_dimensions)

    # Get the best multiset of all stages
    best_round = None
    best_fitness = None
    for stage in round2best:
        if best_fitness is None or best_fitness < round2best[stage][-2]:
            best_round = stage
            best_fitness = round2best[stage][-2]
    all_accuracies, best_index, best_solution, query2query_answers_dump, fitness, used_syntax_dimensions = \
        round2best[best_round]
    return all_accuracies, best_index, best_solution, query2query_answers_dump, used_syntax_dimensions


def save_results(all_accuracies, args, best_index, best_solution, end_time, query2query_answers_dump, start_time,
                 suffix, target_record, used_syntax_dimensions):
    # Print the execution time and the accuracy of the best attack
    print('The total execution time was', end_time - start_time, 'seconds')
    print('The best-performing attack has accuracy of', all_accuracies[best_index])

    # Save the results to the filesystem
    results = {
        'all_accuracies': all_accuracies,
        'best_solution': best_solution,
        'best_index': best_index,
        'best_accuracy': all_accuracies[best_index],
        'time': end_time - start_time,
        'args': args,
        'query2query_answers': query2query_answers_dump,
        'target_record': target_record,
        'used_syntax_dimensions': [] if args.only_limited_syntax else used_syntax_dimensions,
    }
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(f'{args.output_dir}/qc_{suffix}', 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    main()
