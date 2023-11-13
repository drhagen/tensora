__all__ = ["parse_format"]

from parsita import (
    Failure,
    ParseError,
    ParserContext,
    Result,
    StringReader,
    Success,
    eof,
    lit,
    reg,
    rep,
)
from parsita.util import constant

from .format import Format, Mode


def make_format_with_orderings(dims):
    modes = []
    orderings = []
    for mode, ordering in dims:
        modes.append(mode)
        orderings.append(ordering)
    return Format(tuple(modes), tuple(orderings))


class FormatTextParsers(ParserContext):
    integer = reg(r"[0-9]+") > int
    dense = lit("d") > constant(Mode.dense)
    compressed = lit("s") > constant(Mode.compressed)
    mode = dense | compressed

    # Use eof to ensure each parser goes to end
    format_without_orderings = rep(mode) << eof > (
        lambda modes: Format(tuple(modes), tuple(range(len(modes))))
    )
    format_with_orderings = rep(mode & integer) << eof > make_format_with_orderings

    format = format_without_orderings | format_with_orderings


def parse_format(format: str) -> Result[Format]:
    parse_result = FormatTextParsers.format.parse(format)
    if isinstance(parse_result, Failure):
        return parse_result
    elif isinstance(parse_result, Success):
        parse_value = parse_result.unwrap()
        if set(range(parse_value.order)) != set(parse_value.ordering):
            return Failure(
                ParseError(
                    StringReader(format),
                    f"format ordering as some order of the set {set(range(parse_value.order))}",
                )
            )
        else:
            return parse_result
