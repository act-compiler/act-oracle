"""API file generator for Oracle"""

from taidl.template import init_templates, templates, generate_code, write_file, indent_code


def generate_semantic_init(data_models):
    """Generate semantic initialization code"""
    init_templates()
    template_semantic_init = templates["SEMANTIC_INIT"]
    template_semantic_counter = templates["SEMANTIC_COUNTER"]
    template_prologue_init = templates["PROLOGUE_INIT"]

    counters = ""
    prologue = ""
    for model in data_models:
        if (model.var_name == 'd0'):
            continue
        dimensions = model.array_dim_str.replace("'", "")
        mapping = {
            "var_name": model.var_name,
            "var_type": model.var_type,
            "var_dim": dimensions,
            "var_num": model.num_dim_str
        }
        counters += generate_code(template_semantic_counter, mapping)
        prologue += generate_code(template_prologue_init, mapping)

    counters = indent_code(counters, level=2)

    output = generate_code(template_semantic_init, {
        "custom_counters": counters,
        "custom_prologue": prologue
    })
    return output


def generate_api_file(oracle_gen_dir: str, accelerator_name: str, instructions,
                      constants, state, data_models) -> None:
    """Generate the api.py file for oracle"""
    init_templates()

    semantic_init = generate_semantic_init(data_models)

    consts = ""
    for constant in constants:
        consts += "'" + constant.const_name + "': " + str(constant.value) + ",\n"

    state_str = ""
    for constant in state:
        state_str += "'" + constant.const_name + "': " + str(constant.value) + ",\n"

    output = generate_code(templates["API_FILE"], {
        "constants": consts,
        "state": state_str,
        "semantic_init": semantic_init,
        "API_NAME": accelerator_name
    })

    for instruction in instructions:
        output += instruction.generate_api_function()

    write_file(output, oracle_gen_dir + "/api.py")
