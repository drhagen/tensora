from tensora import Mode, Format
from tensora.format import parse_format
from tensora.iteration_graph.iteration_graph import *
from tensora.iteration_graph.merge_lattice import *
from tensora.iteration_graph.identifiable_expression import ast
from tensora.iteration_graph.iteration_graph_to_c_code import iteration_graph_to_c_code, TensorOutput, generate_c_code, \
    KernelType

# def test_temp():
#     method = tensor_method('A(i,j) = B(i,k) * C(k,j)', dict(B='ds', C='ds'), 'dd')
#
from tensora.iteration_graph.problem import Problem


def pf(text: str):
    return parse_format(text).or_die()


# def test_code():
#     assignment = parse_assignment('a(i,j) = b(i,j) * c(i,j)').or_die()
#     code = generate_c_code(assignment, dict(b=pf('ss'), c=pf('ss')), pf('ss'))
#     print(code)


# def test_iteration():
#     assignment = parse_assignment('a(i,j) = b(i,j) * c(i,j)').or_die()
#     generate_iteration_graphs(assignment, dict(b=pf('ss'), c=pf('ss')), pf('ss'))

# def test_temp():
#     from tensora import Tensor
#     a = Tensor.from_lol([[0,1,2],[0,0,3]], format='ss')
#     b = Tensor.from_lol([[1,0,0],[0,2,0]], format='ss')
#     from tensora import evaluate
#     c = evaluate('out = a * b', 'ss', a=a, b=b)

def test_multiply():
    # a(i,j) = b(i,j) * c(i,j); a=ss, b=ss, c=ss
    a = ast.Tensor(TensorLeaf('a', 0), ('i', 'j'), (Mode.compressed, Mode.compressed))
    b = ast.Tensor(TensorLeaf('b', 0), ('i', 'j'), (Mode.compressed, Mode.compressed))
    c = ast.Tensor(TensorLeaf('c', 0), ('i', 'j'), (Mode.compressed, Mode.compressed))

    expression = ast.Multiply(b, c)
    assignment = ast.Assignment(a, expression)

    format = Format((Mode.compressed, Mode.compressed), (0, 1))
    problem = Problem(assignment, {'b': format, 'c': format}, format)

    algo = IterationVariable(
        index_variable='i',
        output=LatticeLeaf(a, 0),
        lattice=LatticeDisjunction(
            LatticeLeaf(b, 0),
            LatticeLeaf(c, 0),
        ),
        next=IterationVariable(
            index_variable='j',
            output=LatticeLeaf(a, 1),
            lattice=LatticeDisjunction(
                LatticeLeaf(b, 1),
                LatticeLeaf(c, 1),
            ),
            next=TerminalExpression(expression),
        )
    )

    print(generate_c_code(problem, algo, KernelType.evaluate).source())


def test_matrix_vector_product():
    # a(i) = b(i,j) * c(j); a=d, b=ds, c=d
    a = ast.Tensor(TensorLeaf('a', 0), ('i',), (Mode.dense,))
    b = ast.Tensor(TensorLeaf('b', 0), ('i', 'j'), (Mode.dense, Mode.compressed))
    c = ast.Tensor(TensorLeaf('c', 0), ('j',), (Mode.dense,))

    expression = ast.Multiply(b, c)
    assignment = ast.Assignment(a, expression)

    format = Format(a.modes, (0,))
    problem = Problem(assignment, {'b': format, 'c': format}, format)

    algo = IterationVariable(
        index_variable='i',
        output=LatticeLeaf(a, 0),
        lattice=LatticeLeaf(b, 0),
        next=Contract(
            next=IterationVariable(
                index_variable='j',
                output=None,
                lattice=LatticeDisjunction(
                    LatticeLeaf(b, 1),
                    LatticeLeaf(c, 0),
                ),
                next=TerminalExpression(expression),
            )
        )
    )

    print(generate_c_code(problem, algo, KernelType.evaluate).source())


def test_add():
    # a(i,j) = b(i,j) + c(i,j); a=ss, b=ss, c=ss
    a = ast.Tensor(TensorLeaf('a', 0), ('i', 'j'), (Mode.compressed, Mode.compressed))
    b = ast.Tensor(TensorLeaf('b', 0), ('i', 'j'), (Mode.compressed, Mode.compressed))
    c = ast.Tensor(TensorLeaf('c', 0), ('i', 'j'), (Mode.compressed, Mode.compressed))

    expression = ast.Add(b, c)
    assignment = ast.Assignment(a, expression)

    format = Format((Mode.compressed, Mode.compressed), (0, 1))
    problem = Problem(assignment, {'b': format, 'c': format}, format)

    algo = IterationVariable(
        index_variable='i',
        output=LatticeLeaf(a, 0),
        lattice=LatticeConjuction(
            LatticeLeaf(b, 0),
            LatticeLeaf(c, 0),
        ),
        next=IterationVariable(
            index_variable='j',
            output=LatticeLeaf(a, 1),
            lattice=LatticeConjuction(
                LatticeLeaf(b, 1),
                LatticeLeaf(c, 1),
            ),
            next=TerminalExpression(expression),
        )
    )

    print(generate_c_code(problem, algo, KernelType.evaluate).source())


def test_contract():
    # a(i) = b(i,j) + c(i,j); a=s, b=ss, c=ss
    a = ast.Tensor(TensorLeaf('a', 0), ('i',), (Mode.compressed,))
    b = ast.Tensor(TensorLeaf('b', 0), ('i', 'j'), (Mode.compressed, Mode.compressed))
    c = ast.Tensor(TensorLeaf('c', 0), ('i', 'j'), (Mode.compressed, Mode.compressed))

    expression = ast.Add(b, c)
    assignment = ast.Assignment(a, expression)

    format = Format((Mode.compressed, Mode.compressed), (0, 1))
    problem = Problem(assignment, {'b': format, 'c': format}, Format((Mode.compressed,), (0,)))

    algo = IterationVariable(
        index_variable='i',
        output=LatticeLeaf(a, 0),
        lattice=LatticeConjuction(
            LatticeLeaf(b, 0),
            LatticeLeaf(c, 0),
        ),
        next=Contract(
            next=IterationVariable(
                index_variable='j',
                output=None,
                lattice=LatticeConjuction(
                    LatticeLeaf(b, 1),
                    LatticeLeaf(c, 1),
                ),
                next=TerminalExpression(expression),
            )
        )
    )

    print(generate_c_code(problem, algo, KernelType.evaluate).source())


def test_add_multiply():
    # f(i) = A0(i) + A1(i,j) * x(j); f=d, A0=d, A1=ds, x=d
    f = ast.Tensor(TensorLeaf('f', 0), ('i',), (Mode.dense,))
    A0 = ast.Tensor(TensorLeaf('A0', 0), ('i',), (Mode.dense,))
    A1 = ast.Tensor(TensorLeaf('A1', 0), ('i', 'j'), (Mode.dense, Mode.compressed))
    x = ast.Tensor(TensorLeaf('x', 0), ('j',), (Mode.dense,))

    expression = ast.Add(A0, ast.Multiply(A1, x))
    assignment = ast.Assignment(f, expression)

    format = Format(f.modes, tuple(range(len(f.modes))))
    problem = Problem(assignment, {'A0': format, 'A1': format, 'x': format}, format)

    algo = IterationVariable(
        index_variable='i',
        output=LatticeLeaf(f, 0),
        lattice=LatticeConjuction(
            LatticeLeaf(A0, 0),
            LatticeLeaf(A1, 0),
        ),
        next=Add(
            name='output',
            terms=[
                TerminalExpression(A0),
                IterationVariable(
                    index_variable='j',
                    output=None,
                    lattice=LatticeDisjunction(
                        LatticeLeaf(A1, 1),
                        LatticeLeaf(x, 0),
                    ),
                    next=TerminalExpression(ast.Multiply(A1, x)),
                ),
            ]
        )
    )

    print(generate_c_code(problem, algo, KernelType.evaluate).source())


def test_rhs():
    # f(i) = A0(i) + A1(i,j) * x(j) + A2(i,k,l)*x(k)*x(l); f=d, A0=d, A1=ds, A2=dss, x=d
    f = ast.Tensor(TensorLeaf('f', 0), ('i',), (Mode.dense,))
    A0 = ast.Tensor(TensorLeaf('A0', 0), ('i',), (Mode.dense,))
    A1 = ast.Tensor(TensorLeaf('A1', 0), ('i', 'j'), (Mode.dense, Mode.compressed))
    A2 = ast.Tensor(TensorLeaf('A2', 0), ('i', 'k', 'l'), (Mode.dense, Mode.compressed, Mode.compressed))
    x1 = ast.Tensor(TensorLeaf('x', 1), ('j',), (Mode.dense,))
    x2 = ast.Tensor(TensorLeaf('x', 2), ('j',), (Mode.dense,))
    x3 = ast.Tensor(TensorLeaf('x', 3), ('j',), (Mode.dense,))

    expression = ast.Add(ast.Add(A0, ast.Multiply(A1, x1)), ast.Multiply(ast.Multiply(A2, x2), x3))
    assignment = ast.Assignment(f, expression)

    format = Format(f.modes, tuple(range(len(f.modes))))
    problem = Problem(assignment, {'A0': format, 'A1': pf('ds'), 'x': format, 'A2': pf('dss')}, format)

    algo = IterationVariable(
        index_variable='i',
        output=LatticeLeaf(f, 0),
        lattice=LatticeConjuction(
            LatticeLeaf(A0, 0),
            LatticeConjuction(
                LatticeLeaf(A1, 0),
                LatticeLeaf(A2, 0),
            )
        ),
        next=Add(
            name='output',
            terms=[
                TerminalExpression(A0),
                IterationVariable(
                    index_variable='j',
                    output=None,
                    lattice=LatticeDisjunction(
                        LatticeLeaf(A1, 1),
                        LatticeLeaf(x1, 0),
                    ),
                    next=TerminalExpression(ast.Multiply(A1, x1)),
                ),
                IterationVariable(
                    index_variable='k',
                    output=None,
                    lattice=LatticeDisjunction(
                        LatticeLeaf(A2, 1),
                        LatticeLeaf(x2, 0),
                    ),
                    next=IterationVariable(
                        index_variable='l',
                        output=None,
                        lattice=LatticeDisjunction(
                            LatticeLeaf(A2, 2),
                            LatticeLeaf(x3, 0),
                        ),
                        next=TerminalExpression(ast.Multiply(ast.Multiply(A2, x2), x3)),
                    ),
                ),
            ]
        )
    )

    print(generate_c_code(problem, algo, KernelType.compute).source())


def test_scratch():
    # a(i,k) = b(i,j) * c(j,k); a=ds, b=ds, c=ds
    a = ast.Tensor(TensorLeaf('a', 0), ('i', 'k'), (Mode.dense, Mode.compressed))
    b = ast.Tensor(TensorLeaf('b', 0), ('i', 'j'), (Mode.dense, Mode.compressed))
    c = ast.Tensor(TensorLeaf('c', 0), ('j', 'k'), (Mode.dense, Mode.compressed))

    expression = ast.Multiply(b, c)
    assignment = ast.Assignment(a, expression)

    format = Format((Mode.dense, Mode.compressed), (0, 1))
    problem = Problem(assignment, {'b': format, 'c': format}, format)

    algo = IterationVariable(
        index_variable='i',
        output=LatticeLeaf(a, 0),
        lattice=LatticeDisjunction(
            LatticeLeaf(b, 0),
            LatticeLeaf(c, 0),
        ),
        next=Scratch(
            layers=[LatticeLeaf(a, 1)],
            next=IterationVariable(
                index_variable='j',
                output=None,
                lattice=LatticeDisjunction(
                    LatticeLeaf(b, 1),
                    LatticeLeaf(c, 0),
                ),
                next=IterationVariable(
                    index_variable='k',
                    output=None,
                    lattice=LatticeLeaf(c, 1),
                    next=TerminalExpression(expression),
                ),
            )
        )
    )

    print(generate_c_code(problem, algo, KernelType.evaluate).source())

# def test_dense():
#     # a(i,j) = b(i,j) + c(i,j); a=ss, b=dd, c=dd
#     algo = IterationVariable(
#         index_variable='i',
#         output_tensor='a',
#         lattice=LatticeConjuction(
#             LatticeLeaf(TensorLeaf('b', 0), 0, Mode.dense),
#             LatticeLeaf(TensorLeaf('c', 0), 0, Mode.dense),
#         ),
#         next=IterationVariable(
#             index_variable='j',
#             output_tensor='a',
#             lattice=LatticeConjuction(
#                 LatticeLeaf(TensorLeaf('b', 0), 1, Mode.dense),
#                 LatticeLeaf(TensorLeaf('c', 0), 1, Mode.dense),
#             ),
#             next=TerminalExpression(ast.Add(
#                 ast.Tensor(TensorLeaf('b', 0), ['i', 'j']),
#                 ast.Tensor(TensorLeaf('c', 0), ['i', 'j']),
#             )),
#         )
#     )
#
#     print(iteration_graph_to_c_code(algo, output=Compressed('a', 1)).source())

# def test_csr():
#     # a(i,j) = b(i,j) + c(i,j); a=ds, b=ds, c=ds
#     algo = IterationVariable(
#         index_variable='i',
#         output=LatticeLeaf(TensorLeaf('b', 0), 1, Mode.dense),
#         lattice=LatticeConjuction(
#             LatticeLeaf(TensorLeaf('b', 0), 0, Mode.dense),
#             LatticeLeaf(TensorLeaf('c', 0), 0, Mode.dense),
#         ),
#         next=IterationVariable(
#             index_variable='j',
#             output=LatticeLeaf(TensorLeaf('b', 0), 1, Mode.compressed),
#             lattice=LatticeConjuction(
#                 LatticeLeaf(TensorLeaf('b', 0), 1, Mode.compressed),
#                 LatticeLeaf(TensorLeaf('c', 0), 1, Mode.compressed),
#             ),
#             next=TerminalExpression(ast.Add(
#                 ast.Tensor(TensorLeaf('b', 0), ['i', 'j']),
#                 ast.Tensor(TensorLeaf('c', 0), ['i', 'j']),
#             )),
#         )
#     )
#
#     print(iteration_graph_to_c_code(algo, output=Compressed('a', 1)).source())
