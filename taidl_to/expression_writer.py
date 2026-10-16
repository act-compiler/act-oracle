"""Python for a value or a condition taidl has parsed, read from a call's attributes and the state"""

from taidl.antlr4 import IDLV2ExprParser, IDLV2ExprVisitor

# The dict of the generated API each reference reads
TABLES = {'@a.': 'attrs', '@c.': 'comp_attrs', '@s.': 'state'}


class ExpressionWriter(IDLV2ExprVisitor):
    """A value as one Python expression, bracketed as the tree groups it."""

    def visitExpression(self, ctx: IDLV2ExprParser.ExpressionContext):
        return self.visit(ctx.additive())

    def chain(self, ctx, operands):
        """A left-associative run of operands, the operators between them."""
        source = self.visit(operands[0])
        for k, operand in enumerate(operands[1:]):
            operator = ctx.getChild(2 * k + 1).getText()
            # The language's / is floor division, as Python's // is
            operator = '//' if operator == '/' else operator
            source = f'({source} {operator} {self.visit(operand)})'
        return source

    def visitAdditive(self, ctx: IDLV2ExprParser.AdditiveContext):
        return self.chain(ctx, ctx.multiplicative())

    def visitMultiplicative(self, ctx: IDLV2ExprParser.MultiplicativeContext):
        return self.chain(ctx, ctx.unary())

    def visitUnary(self, ctx: IDLV2ExprParser.UnaryContext):
        if ctx.MINUS() is None:
            return self.visit(ctx.atom())
        return f'(-{self.visit(ctx.unary())})'

    def visitAtom(self, ctx: IDLV2ExprParser.AtomContext):
        if ctx.INT() is not None:
            return ctx.INT().getText()
        return self.visit(ctx.reference() or ctx.expression())

    def visitReference(self, ctx: IDLV2ExprParser.ReferenceContext):
        return reference(ctx.getText())


class ConditionWriter(ExpressionWriter):
    """A condition as one Python expression over `read`,
    which maps each reference's text to its value."""

    def visitCondition(self, ctx: IDLV2ExprParser.ConditionContext):
        return ' or '.join(self.visit(conjunction) for conjunction in ctx.conjunction())

    def visitConjunction(self, ctx: IDLV2ExprParser.ConjunctionContext):
        return ' and '.join(self.visit(comparison) for comparison in ctx.comparison())

    def visitComparison(self, ctx: IDLV2ExprParser.ComparisonContext):
        # A bracketed condition keeps its brackets
        if ctx.condition() is not None:
            return f'({self.visit(ctx.condition())})'
        left, right = ctx.additive()
        return f'{self.visit(left)} {ctx.compare().getText()} {self.visit(right)}'

    def visitReference(self, ctx: IDLV2ExprParser.ReferenceContext):
        return f'read[{ctx.getText()!r}]'


class References(IDLV2ExprVisitor):
    """The references a value reads, as written."""

    def defaultResult(self):
        return []

    def aggregateResult(self, aggregate, nextResult):
        return aggregate + nextResult

    def visitReference(self, ctx: IDLV2ExprParser.ReferenceContext):
        return [ctx.getText()]


def reference(text: str) -> str:
    """The Python that reads a reference from its dict."""
    return f"{TABLES[text[:3]]}['{text[3:]}']"


def condition(tree: IDLV2ExprParser.ConditionContext) -> tuple[str, str]:
    """The Python of a condition over `read`, and of the dict `read`
    from each reference's text to its value."""
    read = ', '.join(
        f'{text!r}: {reference(text)}' for text in dict.fromkeys(References().visit(tree))
    )
    return ConditionWriter().visit(tree), f'{{{read}}}'


def python(tree: IDLV2ExprParser.ExpressionContext) -> str:
    """The Python that computes a value."""
    return ExpressionWriter().visit(tree)


def held(tree, text: str, instruction: str) -> str:
    """The Python that computes a value, held to one value over its loop if it reads an attribute."""
    read = [ref[3:] for ref in References().visit(tree) if ref.startswith('@a.')]
    if not read:
        return python(tree)
    given = ', '.join(f"'{name}': values['{name}']" for name in dict.fromkeys(read))
    return f'fixed({instruction!r}, {text!r}, {python(tree)}, {{{given}}})'


def integer(ctx, expressions, instruction=None) -> str:
    """An integer as Python: a literal's value, or an expression evaluated at each call and,
    given its instruction, held to one value over its loop."""
    if ctx.EXPRESSION():
        tree = expressions[ctx.EXPRESSION()]
        return python(tree) if instruction is None else held(tree, ctx.getText(), instruction)
    # Drop a literal's leading zeros, which Python refuses
    return str(int(ctx.getText()))
