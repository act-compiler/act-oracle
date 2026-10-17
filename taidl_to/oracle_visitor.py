"""ANTLR4-based visitor turning TAIDL semantics into the code that traces their StableHLO"""

import re

from taidl.antlr4 import IDLV2Parser, IDLV2Visitor

from .expression_writer import python
from .generic.utils import MLIR_ELEM_TYPE, mlir_literal
from .shapes import declared, disagreement
from .template import generate_code, templates

# Element types as a semantics spells them
FLOATS = ('f64', 'f32', 'f16', 'bf16')
SIGNED = ('s8', 's16', 's32', 's64')
INTEGERS = ('u8', 'u16', 'u32', 'u64', *SIGNED)
EVERY = (*FLOATS, *INTEGERS, 'pred')

# Each word's element types, at its first operand or result
TYPES = {
    'reshape': EVERY,
    'convert': EVERY,
    'copy': EVERY,
    'bitcast_convert': EVERY,
    'exponential': FLOATS,
    'transpose': EVERY,
    'add': EVERY,
    'subtract': FLOATS + INTEGERS,
    'multiply': EVERY,
    'divide': FLOATS + INTEGERS,
    'broadcast': EVERY,
    'maximum': EVERY,
    'minimum': EVERY,
    'xor': (*INTEGERS, 'pred'),
    'dot': EVERY,
    'reduce_add': EVERY,
    'constant': EVERY,
    'iota': FLOATS + INTEGERS,
    'abs': (*FLOATS, *SIGNED),
    'negate': FLOATS + INTEGERS,
    'sign': (*FLOATS, *SIGNED),
    'mulhi': INTEGERS,
    'remainder': FLOATS + INTEGERS,
    'power': FLOATS + INTEGERS,
    'sqrt': FLOATS,
    'rsqrt': FLOATS,
    'cbrt': FLOATS,
    'exponential_minus_one': FLOATS,
    'log': FLOATS,
    'log_plus_one': FLOATS,
    'logistic': FLOATS,
    'erf': FLOATS,
    'sine': FLOATS,
    'cosine': FLOATS,
    'tan': FLOATS,
    'asin': FLOATS,
    'acos': FLOATS,
    'atan2': FLOATS,
    'sinh': FLOATS,
    'cosh': FLOATS,
    'tanh': FLOATS,
    'asinh': FLOATS,
    'acosh': FLOATS,
    'atanh': FLOATS,
    'floor': FLOATS,
    'ceil': FLOATS,
    'round_nearest_afz': FLOATS,
    'round_nearest_even': FLOATS,
    'and': (*INTEGERS, 'pred'),
    'or': (*INTEGERS, 'pred'),
    'not': (*INTEGERS, 'pred'),
    'shift_left': INTEGERS,
    'shift_right_logical': INTEGERS,
    'shift_right_arithmetic': INTEGERS,
    'popcnt': INTEGERS,
    'count_leading_zeros': INTEGERS,
    'is_finite': FLOATS,
    'reduce_precision': FLOATS,
}

# The words StableHLO lacks, which CHLO holds
CHLO = ('mulhi', 'erf', 'asin', 'acos', 'sinh', 'cosh', 'asinh', 'acosh', 'atanh')


def generic_attributes(word, attributes):
    """A word's attributes as StableHLO names them, a Python dict evaluated at each call;
    a quoted value is MLIR text"""
    if word == 'reduce_precision':
        bits = ('exponent_bits', 'mantissa_bits')
        given = {name: repr(f'{attributes[name]} : i32') for name in bits}
    else:
        # Any other word's attributes have StableHLO's names
        given = attributes
    return '{' + ', '.join(f"'{name}': {value}" for name, value in given.items()) + '}'


class OracleVisitor(IDLV2Visitor):
    def __init__(self, instruction, expressions) -> None:
        super().__init__()
        # The instruction whose semantics this visits, for a refusal
        self.instruction = instruction
        # Each backtick expression's tree, by its node
        self.expressions = expressions
        # Each value's type and dims, by name
        self.values = {}
        self.instruction_lines = []

        # Each word's template, the generic one where none is named
        self.op_to_template = {
            'reshape': 'RESHAPE',
            'convert': 'CONVERT',
            'copy': 'COPY',
            'bitcast_convert': 'BITCAST_CONVERT',
            'transpose': 'TRANSPOSE',
            'broadcast': 'BROADCAST',
            'dot': 'DOT',
            'reduce_add': 'REDUCE_ADD',
            'constant': 'CONSTANT',
            'iota': 'IOTA',
        }

    def visitInstruction(self, ctx: IDLV2Parser.InstructionContext):
        lhs_name = ctx.IDENTIFIER().getText().lstrip('%')
        result_type_info = self.visit(ctx.result_type())
        lhs_type, lhs_shape = result_type_info
        op_name = ctx.OPERATION().getText()
        self.values[lhs_name] = (lhs_type, declared(ctx.result_type()))

        if op_name == 'parameter':
            return (lhs_name, lhs_type, lhs_shape)

        if op_name not in TYPES:
            raise NotImplementedError(
                f"{self.instruction}: operation '{op_name}' is not supported by the oracle"
            )

        operand_names = []
        if op_name == 'constant':
            operand_names.append(self.constant_value(ctx.operands().operand(0).value(), lhs_type))
        elif ctx.operands():
            operand_info = self.visit(ctx.operands())
            for operand_value in operand_info:
                operand_names.append(operand_value.lstrip('%'))

        # A word at a type it lacks, constant literals aside
        values = [] if op_name == 'constant' else operand_names
        typ = self.values[values[0]][0] if values else lhs_type
        if typ not in TYPES[op_name]:
            raise NotImplementedError(
                f"{self.instruction}: operation '{op_name}' at type '{typ}' "
                'is not supported by the oracle'
            )

        attributes = {}
        if ctx.attributes():
            attribute_info = self.visit(ctx.attributes())
            for attr_name, attr_value in attribute_info:
                attributes[attr_name] = attr_value

        # Refuse a line whose shapes or attributes disagree
        operands = [(f"'%{name}'", self.values[name]) for name in values]
        reason = disagreement(op_name, self.values[lhs_name], operands, attributes)
        if reason:
            raise ValueError(f"{self.instruction}: {op_name} of '%{lhs_name}' {reason}")

        self.generate_from_template(
            op_name, lhs_name, lhs_type, lhs_shape, operand_names, attributes
        )

        return (lhs_name, lhs_type, lhs_shape)

    def generate_from_template(
        self, op_name, lhs_name, lhs_type, lhs_shape, operand_names, attributes
    ):
        template_name = self.op_to_template.get(op_name, 'GENERIC')
        mapping = {'lhs': f"'{lhs_name}'", 'type': f"'{lhs_type}'", 'size': f'[{lhs_shape}]'}

        if template_name in ['RESHAPE', 'CONVERT', 'COPY', 'BITCAST_CONVERT']:
            mapping['in'] = f"'{operand_names[0]}'"

        elif template_name == 'GENERIC':
            dialect = 'chlo' if op_name in CHLO else 'stablehlo'
            mapping['op'] = f"'{dialect}.{op_name}'"
            mapping['operands'] = repr(operand_names)
            mapping['attributes'] = generic_attributes(op_name, attributes)

        elif template_name == 'BROADCAST':
            mapping['A'] = f"'{operand_names[0]}'"
            # broadcast maps operand dims by position; keep the spec's order
            mapping['dims'] = ', '.join(str(d) for d in attributes['dimensions'])

        elif template_name == 'TRANSPOSE':
            mapping['in'] = f"'{operand_names[0]}'"
            mapping['dims'] = ', '.join(str(d) for d in attributes['dimensions'])

        elif template_name == 'DOT':
            mapping['A'] = f"'{operand_names[0]}'"
            mapping['B'] = f"'{operand_names[1]}'"

            # dot pairs dimension lists by position; keep the spec's order
            def format_dims(value):
                return "'{" + ', '.join(str(v) for v in value) + "}'"

            mapping['lb'] = format_dims(attributes.get('lhs_batch_dims', []))
            mapping['lc'] = format_dims(attributes['lhs_contracting_dims'])
            mapping['rb'] = format_dims(attributes.get('rhs_batch_dims', []))
            mapping['rc'] = format_dims(attributes['rhs_contracting_dims'])

        elif template_name == 'REDUCE_ADD':
            mapping['A'] = f"'{operand_names[0]}'"
            mapping['dims'] = '{' + ', '.join(str(d) for d in attributes['dimensions']) + '}'
            template_name = 'REDUCE'

        elif template_name == 'CONSTANT':
            mapping['value'] = operand_names[0]
            mapping['instruction'] = repr(self.instruction)

        elif template_name == 'IOTA':
            mapping['dimension'] = attributes['iota_dimension']

        template_code = generate_code(templates[template_name], mapping)

        for line in template_code.strip().split('\n'):
            self.instruction_lines.append(line)

    def constant_value(self, value, typ):
        """A constant's literal as Python, checked against typ here; an expression at the call"""
        text = value.getText()
        if value.EXPRESSION():
            return python(self.expressions[value.EXPRESSION()])
        if value.IDENTIFIER():
            literal = text == 'true'
        elif re.fullmatch(r'-?[0-9]+', text):
            literal = int(text)
        else:
            literal = float(text)
        try:
            mlir_literal(literal, typ, text)
        except (TypeError, ValueError) as error:
            raise type(error)(f'{self.instruction}: constant {error}') from None
        # The spec's own text, which inf and nan need
        return f"float('{text}')" if isinstance(literal, float) else repr(literal)

    def visitAttribute(self, ctx: IDLV2Parser.AttributeContext):
        attr_name = ctx.name().getText()
        attr_value = self.visit(ctx.attribute_value())
        return (attr_name, attr_value)

    def visitAttributes(self, ctx: IDLV2Parser.AttributesContext):
        attributes_list = []
        for attribute_ctx in ctx.attribute():
            attr_info = self.visit(attribute_ctx)
            attributes_list.append(attr_info)
        return attributes_list

    def visitAttribute_value(self, ctx: IDLV2Parser.Attribute_valueContext):
        if ctx.brace_list():
            return self.visit(ctx.brace_list())
        value = ctx.value()
        if value.EXPRESSION():
            # A scalar attribute lowers from a literal alone
            name = ctx.parentCtx.name().getText()
            raise NotImplementedError(
                f"{self.instruction}: an expression in '{name}' is not supported by the oracle"
            )
        # Drop an integer's leading zeros, which Python refuses
        text = value.getText()
        return str(int(text)) if re.fullmatch(r'-?[0-9]+', text) else text

    def visitBrace_list(self, ctx: IDLV2Parser.Brace_listContext):
        values = []
        for value_ctx in ctx.value():
            if value_ctx.EXPRESSION():
                # The oracle lowers a list of integers alone
                name = ctx.parentCtx.parentCtx.name().getText()
                raise NotImplementedError(
                    f"{self.instruction}: an expression in '{name}' is not supported by the oracle"
                )
            values.append(value_ctx.getText())
        return values

    def visitResult_type(self, ctx: IDLV2Parser.Result_typeContext):
        # Refuse by name a type the oracle does not lower
        typ = ctx.TYPE().getText()
        if typ not in MLIR_ELEM_TYPE:
            raise NotImplementedError(
                f"{self.instruction}: type '{typ}' is not supported by the oracle"
            )
        # A rank-0 result has no shape
        if ctx.shape() is None:
            return (ctx.TYPE().getText(), '')
        shape_info = self.visit(ctx.shape())
        shape_dims, shape_type = shape_info
        return (shape_type, shape_dims)

    def visitShape(self, ctx: IDLV2Parser.ShapeContext):
        # An expression as Python evaluated at each call
        dims = [
            python(self.expressions[dim.EXPRESSION()]) if dim.EXPRESSION() else dim.INT().getText()
            for dim in ctx.dim()
        ]
        typ = ctx.parentCtx.TYPE().getText()
        return (', '.join(dims), typ)

    def visitOperands(self, ctx: IDLV2Parser.OperandsContext):
        operand_list = []
        for operand_ctx in ctx.operand():
            operand_info = self.visit(operand_ctx)
            operand_list.append(operand_info)
        return operand_list

    def visitOperand(self, ctx: IDLV2Parser.OperandContext):
        # A lowered word's every operand is a value's name
        return ctx.value().getText()
