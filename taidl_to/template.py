"""Template processing and code generation utilities for TAIDL-TO"""

import io
import keyword
import os
import re
import tokenize

templates = {}

TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

# A generated line's columns, as ruff lays out the tree
WIDTH = 100


def init_templates():
    filepath = os.path.join(TEMPLATES_DIR, 'templates.txt')
    if templates:
        return
    with open(filepath, 'r') as file:
        content = file.read()
    for section in content.split('### TEMPLATE: '):
        if section.strip():
            header, body = section.split(' ###\n', 1)
            # Each template ends in one newline
            templates[header.strip()] = body.rstrip('\n') + '\n'


def generate_code(template: str, inputs):
    def lines(match):
        # A placeholder alone on a line takes the value's lines
        if match[1] not in inputs:
            return match[0]
        value = inputs[match[1]]
        return value.rstrip('\n') + '\n' if value.strip() else ''

    temp = re.sub(r'^\{\{(\w+)\}\}\n', lines, template, flags=re.MULTILINE)
    for key, value in inputs.items():
        temp = temp.replace(f'{{{{{key}}}}}', value)
    return temp


def write_file(code: str, filename: str):
    with open(filename, 'w') as file:
        file.write(code)


def indent_code(code: str, level=1):
    indentation = '    ' * level
    return '\n'.join(indentation + line if line.strip() else line for line in code.split('\n'))


def dict_text(entries: str) -> str:
    """A dict display of entry lines, each ending in a comma, or {} for none"""
    return '{\n' + indent_code(entries) + '}' if entries else '{}'


def layout(code: str) -> str:
    """The code with each line past WIDTH columns split as ruff splits it"""
    return '\n'.join(split_line(line) for line in code.split('\n'))


def split_line(line: str) -> str:
    """A line split at its last bracket pair, its contents on one line or one item a line"""
    pair = last_pair(line) if len(line) > WIDTH else None
    if pair is None or pair[2] == 'subscript':
        return line
    start, end, kind = pair
    indent = line[: len(line) - len(line.lstrip())]
    inner = indent + '    '
    head, body, tail = line[: start + 1], line[start + 1 : end].strip(), indent + line[end:]
    items = top_level_items(body)
    # A lone item keeps one line, but a def's parameter
    lone = len(items) == 1 and not line.lstrip().startswith('def ')
    # So do a call's arguments that fit
    fits = kind == 'call' and len(items) > 1 and len(inner + body) <= WIDTH
    if lone or fits:
        return '\n'.join([head, split_line(inner + body), tail])
    # Else each item takes a line and a comma
    return '\n'.join([head, *(split_line(inner + item + ',') for item in items), tail])


def tokens(text: str):
    """The tokens of one line outside f-strings"""
    depth = 0
    for token in tokenize.generate_tokens(io.StringIO(text + '\n').readline):
        if token.type == tokenize.FSTRING_START:
            depth += 1
        elif token.type == tokenize.FSTRING_END:
            depth -= 1
        elif not depth and token.type not in (tokenize.INDENT, tokenize.DEDENT):
            yield token


def last_pair(line: str):
    """The columns of the line's last outermost bracket pair, and what it brackets"""
    depth, before, pair = 0, None, None
    for token in tokens(line):
        if token.type == tokenize.OP and token.string in '([{':
            if not depth:
                # A name or a bracket before it calls or subscripts
                follows = before is not None and (
                    (before.type == tokenize.NAME and not keyword.iskeyword(before.string))
                    or before.string in ')]'
                    or before.type == tokenize.STRING
                )
                kind = {'(': 'call', '[': 'subscript'}.get(token.string) if follows else None
                opened = (token.start[1], kind or 'literal')
            depth += 1
        elif token.type == tokenize.OP and token.string in ')]}':
            depth -= 1
            if not depth:
                pair = (opened[0], token.start[1], opened[1])
        before = token
    return pair


def top_level_items(body: str) -> list[str]:
    """The body's items between its outermost commas"""
    depth, cuts = 0, [-1]
    for token in tokens(body):
        if token.type == tokenize.OP and token.string in '([{':
            depth += 1
        elif token.type == tokenize.OP and token.string in ')]}':
            depth -= 1
        elif token.type == tokenize.OP and token.string == ',' and not depth:
            cuts.append(token.start[1])
    items = [body[a + 1 : b].strip() for a, b in zip(cuts, cuts[1:] + [len(body)])]
    return [item for item in items if item]
