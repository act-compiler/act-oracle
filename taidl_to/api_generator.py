"""API file generator for Oracle"""

from .instruction_codegen import generate_api_function
from .template import dict_text, generate_code, indent_code, init_templates, layout, templates


def generate_semantic_init(data_models):
    """Generate semantic initialization code"""
    init_templates()
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
    init_templates()

    semantic_init = generate_semantic_init(data_models)

    state_str = ''
    for variable in state:
        state_str += "'" + variable.name + "': " + str(variable.value) + ',\n'

    output = generate_code(
        templates['API_FILE'],
        {'state': dict_text(state_str), 'semantic_init': semantic_init},
    )

    for instruction in instructions:
        output += generate_api_function(instruction)

    return output
