import dataclasses
import typing as t
from pathlib import Path

from sly import (
    Lexer,
    Parser,
)

ENCODINGS = {'UTF8'}


@dataclasses.dataclass
class Annotation:
    speaker: str
    text: t.AnyStr
    meta: dict[str, t.AnyStr]


@dataclasses.dataclass
class CHAFileAST:
    header: dict[str, t.Any]
    annotations: list[Annotation]

    def filter(self, speaker: str) -> t.Iterator[Annotation]:
        yield from (x for x in self.annotations if x.speaker == speaker)


class CHALexer(Lexer):
    """ Lexer for CHA files """
    tokens = {
        ENCODING, BEGIN, END,  # noqa: sly syntaxt
        HEADER_NAME, ANNOT_NAME, META_NAME,  # noqa: sly syntaxt
        VALUE_STRING  # noqa: sly syntaxt
    }
    ignore = ' \t'
    literals = {}

    ENCODING = '@UTF8'
    BEGIN = r"@Begin"
    END = r"@End"
    HEADER_NAME = r"@[A-Za-z0-9]+:"
    ANNOT_NAME = r"\*[A-Za-z0-9]+:"
    META_NAME = r"%[A-Za-z0-9]+:"
    VALUE_STRING = r'[^@\*%\n]+'

    @_(r'\n+')  # noqa: sly syntax
    def newline(self, t):
        self.lineno += t.value.count('\n')

    @staticmethod
    def find_column(text, token):
        last_cr = text.rfind('\n', 0, token.index)
        if last_cr < 0:
            last_cr = 0
        column = (token.index - last_cr) + 1
        return column

    def error(self, t):
        print(f"Illegal character {t.value[0]} @ {self.index}:{self.lineno}")
        self.index += 1


"""
statement := HEADER | HEADER_VALUE | TRANSCRIPT | ANNOTATION
HEADER := NAME
HEADER_VALUE := NAME ':' VALUE
TRANSCRIPT := NAME ':' VALUE
ANNOTATION := NAME ':' VALUE
NAME := STRING
NAME := STRING

"""


class CHAParser(Parser):
    tokens = CHALexer.tokens
    debugfile = 'parser.out'

    def __init__(self):
        self.parsed_data = CHAFileAST(header={}, annotations=[])
        self._current_annotation = None

    @_(r'HEADER_NAME VALUE_STRING')  # noqa: sly syntaxt
    def statement(self, line):
        name = line.HEADER_NAME.replace('@', '').replace(':', '')
        value = line.VALUE_STRING

        if name in self.parsed_data.header:
            self.parsed_data.header[name] += f" {value}"
        else:
            self.parsed_data.header[name] = value

    @_(r'ANNOT_NAME VALUE_STRING')  # noqa: sly syntaxt
    def statement(self, line):
        name = line.ANNOT_NAME.replace('*', '').replace(':', '')
        value = line.VALUE_STRING

        if self._current_annotation is not None:
            self.parsed_data.annotations.append(self._current_annotation)
            self._current_annotation = None

        self._current_annotation = Annotation(
            speaker=name,
            text=value,
            meta={}
        )

    @_(r'META_NAME VALUE_STRING')  # noqa: sly syntaxt
    def statement(self, line):
        name = line.META_NAME.replace('%', '').replace(':', '')
        value = line.VALUE_STRING

        if self._current_annotation is None:
            raise ValueError('No annotation for meta')

        self._current_annotation.meta[name] = value


    @_(r'ENCODING')  # noqa: sly syntaxt
    def statement(self, line):
        self.parsed_data.encoding = line.ENCODING.replace('@', '')

    @_(r'BEGIN')  # noqa: sly syntaxt
    def statement(self, line):
        pass

    @_(r'END')  # noqa: sly syntaxt
    def statement(self, line):
        pass

    @_(r'HEADER_NAME')  # noqa: sly syntaxt
    def statement(self, line):
        name = line.HEADER_NAME.replace('@', '').replace(':', '')
        if name in ENCODINGS:
            self.parsed_data.header['encoding'] = name
        print(f"{name}")

    def error(self, p):
        if p:
            print("Syntax error at token", p.type)
            print(p)
            # Just discard the token and tell the parser it's okay.
            self.errok()
        else:
            print("Syntax error at EOF")


if __name__ == '__main__':
    # Testing
    test_data = Path("~/workspace/coml/data/Lexical-benchmark/data/datasets").expanduser()
    lexer = CHALexer()
    parser = CHAParser()
    cha_text = (test_data / 'Bates/Free28/amy28.cha').read_text()
    tokens = lexer.tokenize(cha_text)
    result = parser.parse(tokens)
