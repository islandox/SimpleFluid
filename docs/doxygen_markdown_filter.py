#!/usr/bin/env python3
"""Translate GitHub inline and display math to Doxygen formula commands."""

from pathlib import Path
import re
import sys
from typing import List, Optional, Tuple


FENCE = re.compile(
    r"^(?P<indent> {0,3})(?P<marker>`{3,}|~{3,})"
    r"(?P<info>[^\r\n]*)(?P<eol>\r?\n)?$"
)
DISPLAY_MATH_DELIMITER = re.compile(r"^ {0,3}\$\$[ \t]*(?:\r?\n)?$")


def fence(line: str) -> Optional[Tuple[str, int, str]]:
    match = FENCE.fullmatch(line)
    if match is None:
        return None
    marker = match.group("marker")
    return marker[0], len(marker), match.group("info").strip()


def closes_fence(line: str, marker: str, minimum_length: int) -> bool:
    parsed = fence(line)
    return (
        parsed is not None
        and parsed[0] == marker
        and parsed[1] >= minimum_length
        and not parsed[2]
    )


def escaped(text: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def code_span_end(text: str, start: int) -> Optional[int]:
    marker_end = start
    while marker_end < len(text) and text[marker_end] == "`":
        marker_end += 1
    marker_length = marker_end - start

    search = marker_end
    while search < len(text):
        candidate = text.find("`", search)
        if candidate < 0:
            return None
        candidate_end = candidate
        while candidate_end < len(text) and text[candidate_end] == "`":
            candidate_end += 1
        if candidate_end - candidate == marker_length:
            return candidate_end
        search = candidate_end
    return None


def inline_math_end(text: str, start: int) -> Optional[int]:
    search = start + 1
    while search < len(text):
        candidate = text.find("$", search)
        if candidate < 0:
            return None
        if (
            not escaped(text, candidate)
            and text[candidate - 1] not in " \t\r\n"
            and (candidate + 1 == len(text) or text[candidate + 1] != "$")
        ):
            return candidate
        search = candidate + 1
    return None


def translate_inline_math(line: str) -> str:
    output: List[str] = []
    index = 0

    while index < len(line):
        if line[index] == "`":
            end = code_span_end(line, index)
            if end is None:
                output.append(line[index:])
                break
            output.append(line[index:end])
            index = end
            continue

        if line.startswith("$`", index) and not escaped(line, index):
            end = line.find("`$", index + 2)
            if end > index + 2:
                output.extend(("\\f$", line[index + 2:end], "\\f$"))
                index = end + 2
                continue

        if (
            line[index] == "$"
            and not escaped(line, index)
            and (index == 0 or line[index - 1] != "$")
            and index + 1 < len(line)
            and line[index + 1] not in "$ \t\r\n"
        ):
            end = inline_math_end(line, index)
            if end is not None:
                output.extend(("\\f$", line[index + 1:end], "\\f$"))
                index = end + 1
                continue

        output.append(line[index])
        index += 1

    return "".join(output)


def transform(text: str) -> str:
    output: List[str] = []
    pending_math: Optional[List[str]] = None
    math_fence: Optional[Tuple[str, int]] = None
    ordinary_fence: Optional[Tuple[str, int]] = None

    for line in text.splitlines(keepends=True):
        if ordinary_fence is not None:
            output.append(line)
            if closes_fence(line, *ordinary_fence):
                ordinary_fence = None
            continue

        if pending_math is not None:
            closes_math = (
                DISPLAY_MATH_DELIMITER.fullmatch(line) is not None
                if math_fence is None
                else closes_fence(line, *math_fence)
            )
            if not closes_math:
                pending_math.append(line)
                continue

            body = pending_math[1:]
            output.append("\\f[\n")
            output.extend(body)
            if body and not body[-1].endswith(("\n", "\r")):
                output.append("\n")
            output.append("\\f]\n" if line.endswith(("\n", "\r")) else "\\f]")
            pending_math = None
            math_fence = None
            continue

        if DISPLAY_MATH_DELIMITER.fullmatch(line):
            pending_math = [line]
            math_fence = None
            continue

        parsed = fence(line)
        if parsed is None:
            output.append(translate_inline_math(line))
            continue

        marker, length, info = parsed
        if marker == "`" and info == "math":
            pending_math = [line]
            math_fence = (marker, length)
        else:
            output.append(line)
            ordinary_fence = (marker, length)

    if pending_math is not None:
        output.extend(pending_math)

    return "".join(output)


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} MARKDOWN_FILE", file=sys.stderr)
        return 2

    source = Path(sys.argv[1])
    sys.stdout.write(transform(source.read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
