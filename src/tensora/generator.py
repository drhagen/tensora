from typing import Dict, Optional

from tensora import Format, Mode
from tensora.expression import Assignment
from tensora.expression.ast import Add, Multiply, Tensor, Node

code_source = """\
int evaluate({{#each declaration}}taco_tensor_t *{{this}}{{#unless @last}}, {{/unless}}{{/each}})
{
  int32_t {{target.name}}_capacity = 1;
{{#each target.formats}}
{{#if dense}}
  {{target.name}}_vals_capacity *= {{target.name}}{{@index}}_dimension;
{{#else}}
  int32_t 
{{/if}}
{{/each}}
  {{target.name}}_vals = (double*)malloc(sizeof(double) * {{target.name}}_capacity);
}

  return 0;
"""


def generate_c_code(assignment: Assignment, input_formats: Dict[str, Format], output_format: Format) -> str:
    # This only works for assignments with zero or one layer of addition above zero or one layer of multiplication of
    # tensors. Each multiplication term must contain all output indexes (no broadcasting). Also, all tensors must have
    # the format ds*.
    def flatten_multiplication(node: Node):
        if isinstance(node, Multiply):
            return flatten_multiplication(node.left) + flatten_multiplication(node.right)
        elif isinstance(node, Tensor):
            return [node]
        else:
            raise NotImplementedError

    def flatten_addition(node: Node):
        if isinstance(node, Add):
            return flatten_addition(node.left) + flatten_addition(node.right)
        elif isinstance(node, (Multiply, Tensor)):
            return [flatten_multiplication(node)]
        else:
            raise NotImplementedError

    terms = flatten_addition(assignment.expression)

    target_name = assignment.target.name

    tensor_names = [target_name] + list(input_formats.keys())

    source_lines = [
        f"int evaluate({', '.join(f'taco_tensor_t * {name}' for name in tensor_names)})",
        '{',
    ]

    # Unpack tensors
    for name, format in {target_name: output_format, **input_formats}.items():
        for i, mode in enumerate(format.modes):
            dim_name = f'{name}{i + 1}'
            if mode == Mode.dense:
                source_lines.append(f'  int {dim_name}_dimension = (int)({name}->dimensions[{i}]);')
            elif mode == Mode.compressed:
                source_lines += [
                    f'  int* restrict {dim_name}_pos = (int*)({name}->indices[{i}][0]);',
                    f'  int* restrict {dim_name}_crd = (int*)({name}->indices[{i}][1]);',
                ]
            else:
                raise NotImplementedError
        source_lines.append(f'  double* restrict {name}_vals = (double*)({name}->vals);')

    source_lines.append('')

    # Allocate memory for target
    all_dense = True
    for i, mode in enumerate(output_format.modes):
        dim_name = f'{target_name}{i+1}'
        if mode == Mode.dense:
            pass
        elif mode == Mode.compressed:
            # How pos is handled depends on what the previous modes were
            if all_dense:
                # If the previous dimensions were all dense, then the size of pos in this dimension is fixed
                if i == 0:
                    nnz_string = '2'
                else:
                    nnz_string = f'({" * ".join(f"{target_name}{i_prev}_dimension" for i_prev in range(i))} + 1)'

                source_lines += [
                    f'  {dim_name}_pos = (int32_t*)malloc(sizeof(int32_t) * {nnz_string};',
                    f'  {dim_name}_pos[0] = 0;',
                ]
            else:
                source_lines += [
                    f'  int32_t {dim_name}_pos_size = 1048576;',
                    f'  {dim_name}_pos = (int32_t*)malloc(sizeof(int32_t) * {dim_name}_pos_size);',
                    f'  {dim_name}_pos[0] = 0;',
                ]

            source_lines += [
                f'  int32_t {dim_name}_crd_size = 1048576;',
                f'  {dim_name}_crd = (int32_t*)malloc(sizeof(int32_t) * {dim_name}_crd_size);',
                f'  int32_t {assignment.target.indexes[i]}{target_name} = 0;',
            ]

            all_dense = False
        else:
            raise NotImplementedError

    if all_dense:
        all_dimensions = ' * '.join(f'{target_name}{i}_dimension' for i in range(output_format.order))
        source_lines.append(f"  int32_t {target_name}_capacity = {all_dimensions};")
    else:
        source_lines.append(f'  int32_t {target_name}_capacity = 1048576;')
    source_lines.append(f'  {target_name}_vals = (double*)malloc(sizeof(double) * {target_name}_capacity);')

    # Loop over target dimensions
    outer_index = assignment.target.indexes[0]
    source_lines.append(
        f'  for (int32_t {outer_index} = 0; {outer_index} < {target_name}1_dimension; {outer_index}++) {{'
    )

    source_lines.append('}')
    return '\n'.join(source_lines)
