__all__ = ["parse_format"]

from parsita import ParseError, ParserContext, lit, reg, rep
from parsita.util import constant
from returns import result

from .exceptions import InvalidModeOrderingError
from .format import Format, Mode


def make_format_with_orderings(dims):
    modes = []
    orderings = []
    for mode, ordering in dims:
        modes.append(mode)
        orderings.append(ordering)
    return Format(tuple(modes), tuple(orderings))


class FormatParsers(ParserContext):
    integer = reg(r"[0-9]+") > int
    dense = lit("d") > constant(Mode.dense)
    compressed = lit("s") > constant(Mode.compressed)
    mode = dense | compressed

    format_without_orderings = rep(mode) > (
        lambda modes: Format(tuple(modes), tuple(range(len(modes))))
    )
    format_with_orderings = rep(mode & integer) > make_format_with_orderings

    format = format_without_orderings | format_with_orderings


def parse_format(format: str) -> result.Result[Format, ParseError | InvalidModeOrderingError]:
    try:
        return FormatParsers.format.parse(format)
    except InvalidModeOrderingError as e:
        return result.Failure(e)
