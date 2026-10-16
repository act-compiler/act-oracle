"""API file generator for Oracle"""

import ast
import os
import symtable

from .instruction_codegen import generate_api_function, python_name
from .template import dict_text, generate_code, indent_code, layout, templates

# The decorator, which reads and sets names of api.py
DECORATOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'generic', 'decorator.py')


def generate_semantic_init(data_models):
    """Generate semantic initialization code"""
    template_semantic_init = templates['SEMANTIC_INIT']
    template_semantic_counter = templates['SEMANTIC_COUNTER']
    template_prologue_init = templates['PROLOGUE_INIT']

    counters = ''
    prologue = ''
    for model in data_models:
        if model.var_name == 'd0':
            continue
        dimensions = model.array_dim_str.replace("'", '')
        mapping = {'var_name': model.var_name, 'var_type': model.var_type, 'var_dim': dimensions}
        counters += generate_code(template_semantic_counter, mapping)
        prologue += generate_code(template_prologue_init, mapping)

    counters = indent_code(counters, level=2)

    output = generate_code(
        template_semantic_init, {'custom_counters': counters, 'custom_prologue': prologue}
    )
    return layout(output)


def generate_api_file(instructions, state, data_models) -> str:
    """Generate the text of the oracle's api.py"""
    semantic_init = generate_semantic_init(data_models)

    state_str = ''
    for variable in state:
        state_str += "'" + variable.name + "': " + str(variable.value) + ',\n'

    head = generate_code(
        templates['API_FILE'],
        {'state': dict_text(state_str), 'semantic_init': semantic_init},
    )
    functions = [generate_api_function(instruction) for instruction in instructions]

    # Refuse an instruction named as Python's or api.py's own
    names = [instruction.instruction for instruction in instructions]
    clash = [name for name in names if python_name(name)]
    if not clash:
        reserved = reserved_names(head, functions)
        clash = [name for name in names if name in reserved]
    if clash:
        raise NotImplementedError(
            f"{clash[0]}: instruction name '{clash[0]}' is reserved by the oracle"
        )

    return head + ''.join(functions)


def reserved_names(head: str, functions: list[str]) -> set[str]:
    """What api.py keeps: its head's names, its functions' globals and the decorator's api names."""
    names = {
        symbol.get_name() for symbol in symtable.symtable(head, 'api.py', 'exec').get_symbols()
    }
    tables = [symtable.symtable(head + ''.join(functions), 'api.py', 'exec')]
    while tables:
        table = tables.pop()
        if table.get_type() == 'function':
            names |= {symbol.get_name() for symbol in table.get_symbols() if symbol.is_global()}
        tables += table.get_children()
    with open(DECORATOR) as file:
        tree = ast.parse(file.read())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == 'api'
        ):
            names.add(node.attr)
    return names
