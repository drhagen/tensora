from functools import singledispatch

from ..expression import ast as sugar


def desugar_assignment(assignment: sugar.Assignment):
    all_indexes = set(assignment.index_participants().keys())
    contract_indexes = all_indexes - set(assignment.target.indexes)

    external_indexes = set(assignment.target.indexes)

    desugared_right_hand_side = desugar_expression(assignment.expression, external_indexes)

if __name__ == '__main__':
    from ..expression import parse_assignment
    assignment = parse_assignment('y(i) = A(i,j) * x(j)').or_die()
    desugar_assignment(assignment)
