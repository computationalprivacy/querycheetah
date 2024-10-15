import argparse


class Parser:

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(description='QueryCheetah.')
        parser.add_argument('--dataset_path', type=str, default='datasets')
        parser.add_argument('--dataset_name', type=str, default='adult')
        parser.add_argument('--dataset_filename', type=str, default='final_dataset')
        parser.add_argument('--num_attributes', type=int, default=5)
        parser.add_argument('--dataset_size', type=int, default=8000)
        parser.add_argument('--eval_fraction', type=float, default=0.33333334)
        parser.add_argument('--num_target_qbses', type=int, default=500)
        parser.add_argument('--num_training_qbses', type=int, default=3000)
        parser.add_argument('--num_procs', type=int, default=10)
        parser.add_argument('--num_iterations', type=int, default=5000)
        parser.add_argument('--num_queries', type=int, default=100)
        parser.add_argument('--target_user_index', type=int, default=0)
        parser.add_argument('--repetition', type=int, default=0)
        parser.add_argument('--differentiate_continuous_columns', type=int, default=0)
        parser.add_argument('--num_continuous_attributes', type=int, default=5)
        parser.add_argument('--use_limited_syntax_fast_qbs', type=int, default=0)
        parser.add_argument('--default_seed', type=int, default=0)
        parser.add_argument('--note', type=str, default='')
        parser.add_argument('--output_dir', type=str, default='results')
        parser.add_argument('--use_mitigations', type=int, default=0)
        parser.add_argument('--change_k_queries_at_each_iteration', type=int, default=1)
        parser.add_argument('--attack_type', type=str, default='aia')
        parser.add_argument('--only_limited_syntax', type=int, default=1)
        parser.add_argument('--only_post_hoc', type=int, default=0)
        parser.add_argument('--qbs', type=str, default='diffix')
        args = parser.parse_args()

        print(args)

        Parser.check_validity_of_arguments(args)
        Parser.post_process_arguments(args)
        return args

    @staticmethod
    def check_validity_of_arguments(args):
        assert args.use_mitigations in [0, 1]
        assert args.differentiate_continuous_columns in [0, 1]
        assert args.attack_type in ['aia', 'mia']
        assert args.only_limited_syntax in [0, 1]
        assert args.only_post_hoc in [0, 1]
        assert args.use_limited_syntax_fast_qbs in [0, 1]
        if args.use_limited_syntax_fast_qbs == 1:
            assert args.only_limited_syntax == 1
            assert args.attack_type == 'aia'

    @staticmethod
    def post_process_arguments(args):
        if args.attack_type == 'mia':
            args.dataset_name = f'{args.dataset_name}_with_sensitive'
        # always with the same value
        args.syntax_dimensions = ["use_target_user_values", "use_operator_in", "use_operator_between",
                                  "use_neq_multiple_times"]
        syntax_dimensions2default_values_limited_syntax = {
            'use_target_user_values': 1, 'use_operator_in': 0, 'use_operator_between': 0, 'use_neq_multiple_times': 0,
        }
        for syntax_dimension in args.syntax_dimensions:
            vars(args)[f'{syntax_dimension}'] = syntax_dimensions2default_values_limited_syntax[syntax_dimension]
        args.use_shadow_table = args.use_mitigations
        args.use_isolating_columns = args.use_mitigations
        args.all_uids_for_dynamic_noise = 1 - args.use_mitigations
        args.generic_noise_to_non_comparison_query = args.use_mitigations
