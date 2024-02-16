__all__ = ["deparse_to_taco"]

from functools import singledispatch

from tensora.expression import ast


@singledispatch
def deparse_to_taco_expression(self: ast.Expression) -> str:
    raise NotImplementedError(
        f"deparse_to_taco_expression not implemented for {type(self)}: {self}"
    )


@deparse_to_taco_expression.register(ast.Integer)
def deparse_to_taco_integer(self: ast.Integer) -> str:
    return str(self.value)


@deparse_to_taco_expression.register(ast.Float)
def deparse_to_taco_float(self: ast.Float) -> str:
    return str(self.value)


@deparse_to_taco_expression.register(ast.Tensor)
def deparse_to_taco_tensor(self: ast.Tensor) -> str:
    if len(self.indexes) == 0:
        # Taco represents zero-dimensional tensors as scalars
        return self.name
    else:
        return f"{self.name}({', '.join(self.indexes)})"


@deparse_to_taco_expression.register(ast.Add)
def deparse_to_taco_add(self: ast.Add) -> str:
    return f"{deparse_to_taco_expression(self.left)} + {deparse_to_taco_expression(self.right)}"


@deparse_to_taco_expression.register(ast.Subtract)
def deparse_to_taco_subtract(self: ast.Subtract) -> str:
    return f"{deparse_to_taco_expression(self.left)} - {deparse_to_taco_expression(self.right)}"


@deparse_to_taco_expression.register(ast.Multiply)
def deparse_to_taco_multiply(self: ast.Multiply) -> str:
    left_string = deparse_to_taco_expression(self.left)
    if isinstance(self.left, (ast.Add, ast.Subtract)):
        left_string = f"({left_string})"

    right_string = deparse_to_taco_expression(self.right)
    if isinstance(self.right, (ast.Add, ast.Subtract)):
        right_string = f"({right_string})"

    return f"{left_string} * {right_string}"


def deparse_to_taco(self: ast.Assignment) -> str:
    return f"{deparse_to_taco_expression(self.target)} = {deparse_to_taco_expression(self.expression)}"
