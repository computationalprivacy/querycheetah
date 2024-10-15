import sqlparse


class Parsing:
    @staticmethod
    def get_comparisons(query):
        """
        :param query: the complete SQL query that needs to be parsed
        :return: list of tuples. Each tuple has 3 elements: 1) name of the attribute, 2) comparison operator, and
                 3) value against which it is compared
        """
        comparisons = []
        assert len(sqlparse.split(query)) == 1, 'It is expected that the query passed contains only 1 SQL query command'

        ast = Parsing._get_ast_from_sql_query(query)
        for subtoken in Parsing._generate_comparison_tokens(ast):
            # subtoken.tokens is list of tokens only in the conditions/comparison
            parsed_comparison = Parsing._get_tuple_from_comparison_token(subtoken.tokens)
            comparisons.append(parsed_comparison)

        comparisons.extend(Parsing._generate_comparisons_with_between_keyword(ast))
        return comparisons

    @staticmethod
    def _get_ast_from_sql_query(query):
        parsed = sqlparse.parse(query)[0]
        return parsed.tokens

    @staticmethod
    def _generate_comparison_tokens(ast, max_level=None):
        select_criteria = lambda token: isinstance(token, sqlparse.sql.Comparison)
        where_clause = [token for token in ast if isinstance(token, sqlparse.sql.Where)]
        if len(where_clause) == 0:
            return []
        assert len(where_clause) == 1
        where_clause = where_clause[0]
        return Parsing.get_all_tokens_of_type(where_clause, select_criteria=select_criteria, max_level=max_level)

    @staticmethod
    def _generate_comparisons_with_between_keyword(ast, max_level=None):
        where_clause = [token for token in ast if isinstance(token, sqlparse.sql.Where)]
        if len(where_clause) == 0:
            return []
        assert len(where_clause) == 1
        where_clause = where_clause[0]

        between_conditions = Parsing.get_all_tokens_of_type(where_clause, select_criteria=lambda token: str(
            token.ttype) == 'Token.Keyword' and str(token).lower() == 'between', max_level=max_level,
                                                            get_next_k_tokens=7, get_prev_k_tokens=2)
        parsed_conditions = []
        for between_condition_tokens in between_conditions:
            parsed_conditions.append((str(between_condition_tokens[0]), str(between_condition_tokens[2]),
                                      (str(between_condition_tokens[4]), str(between_condition_tokens[8]))))
        return parsed_conditions

    @staticmethod
    def _get_tuple_from_comparison_token(list_tokens_children_of_comparison):
        attribute_name = None
        comparison_operator = None
        value = None
        for token in list_tokens_children_of_comparison:
            if token.is_whitespace:
                continue
            if isinstance(token, sqlparse.sql.Identifier):
                attribute_name = token.value
            elif str(token.ttype) == 'Token.Operator.Comparison':
                comparison_operator = token.value
            elif str(token.ttype) == 'Token.Literal.Number.Integer' or str(token.ttype) == 'Token.Literal.Number.Float':
                value = token.value
            elif isinstance(token, sqlparse.sql.Parenthesis):
                assert comparison_operator == 'IN'
                value = tuple([token.value for token in token.tokens[1].tokens if
                               str(token.ttype) == 'Token.Literal.Number.Integer' or str(
                                   token.ttype) == 'Token.Literal.Number.Float'])

        return attribute_name, comparison_operator, value

    @staticmethod
    def get_all_tokens_of_type(ancestor_token, select_criteria, level=1, max_level=None, get_next_k_tokens=0,
                               get_prev_k_tokens=0):
        # usually, ancestor_token == where_clause
        if max_level is not None and max_level < level:
            return []
        comparison_tokens = []
        for child_index, token in enumerate(ancestor_token.tokens):
            if select_criteria(token):
                comparison_tokens.append(token if get_next_k_tokens == 0 and get_prev_k_tokens == 0 else (
                ancestor_token.tokens[child_index - get_prev_k_tokens:child_index + get_next_k_tokens]))
            if isinstance(token, sqlparse.sql.Parenthesis):
                comparison_tokens.extend(
                    Parsing.get_all_tokens_of_type(token, select_criteria, level=level + 1, max_level=max_level))
        return comparison_tokens
