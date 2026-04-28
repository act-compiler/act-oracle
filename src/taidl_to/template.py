"""Template processing and code generation utilities for TAIDL-TO"""

import os

templates = {}

_TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


def init_templates(filename="templates.txt"):
    global templates
    filepath = os.path.join(_TEMPLATES_DIR, filename)
    if templates:
        return
    with open(filepath, "r") as file:
        content = file.read()
    for section in content.split("### TEMPLATE: "):
        if section.strip():
            header, body = section.split(" ###\n", 1)
            body = body.replace('tab ', '\t')
            templates[header.strip()] = body


def generate_code(template: str, inputs):
    temp = template
    for key, value in inputs.items():
        temp = temp.replace(f"{{{{{key}}}}}", value)
    return temp


def write_file(code: str, filename: str):
    with open(filename, "w") as file:
        file.write(code)


def indent_code(code: str, level=1):
    indentation = '\t' * level
    return "\n".join(indentation + line if line.strip() else line for line in code.split('\n'))
