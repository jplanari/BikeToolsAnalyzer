"""Month and day names in the rider's language.

`fdate(value, "%a %d %b")` is `value.strftime(...)` with %a %A %b %B translated. t() applies the same to any date
that is a placeholder: `t("Week of {monday:%d %b}", monday=monday)`. The names are ordinary catalog entries
(English text is the key), found by the scanner through mark()."""
import re

from . import mark, tr

WEEKDAYS = [mark("Monday"), mark("Tuesday"), mark("Wednesday"), mark("Thursday"), mark("Friday"), mark("Saturday"),
            mark("Sunday")]
WEEKDAYS_SHORT = [mark("Mon"), mark("Tue"), mark("Wed"), mark("Thu"), mark("Fri"), mark("Sat"), mark("Sun")]
MONTHS = [mark("January"), mark("February"), mark("March"), mark("April"), mark("May"), mark("June"), mark("July"),
          mark("August"), mark("September"), mark("October"), mark("November"), mark("December")]
MONTHS_SHORT = [mark("Jan"), mark("Feb"), mark("Mar"), mark("Apr"), mark("May"), mark("Jun"), mark("Jul"),
                mark("Aug"), mark("Sep"), mark("Oct"), mark("Nov"), mark("Dec")]

_NAMES = re.compile(r"%([aAbB])")


def translate_spec(value, spec):
    """`spec` with %a %A %b %B replaced by the translated names for `value`'s date."""
    def name(m):
        code = m.group(1)
        table = {"a": WEEKDAYS_SHORT, "A": WEEKDAYS, "b": MONTHS_SHORT, "B": MONTHS}[code]
        index = value.weekday() if code in "aA" else value.month - 1
        return tr(table[index]).replace("%", "%%")
    try:
        return _NAMES.sub(name, spec)
    except (ValueError, TypeError, AttributeError):          # NaT and the like: leave the spec alone
        return spec


def fdate(value, spec):
    """Like value.strftime(spec), with month and day names in the current language."""
    return value.strftime(translate_spec(value, spec))


class Dated:
    """A date as a t() placeholder: format specs with %a %b ... come out translated."""
    def __init__(self, value):
        self.value = value

    def __format__(self, spec):
        return format(self.value, translate_spec(self.value, spec)) if spec else str(self.value)

    def __str__(self):
        return str(self.value)
