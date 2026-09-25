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
DISPLAY_MATH_OPEN = re.compile(r"^ {0,3}\\\[[ \t]*(?:\r?\n)?$")
DISPLAY_MATH_CLOSE = re.compile(r"^ {0,3}\\\][ \t]*(?:\r?\n)?$")
DOXYGEN_HEADER_PAGES = {
    "../src/materials/MaterialCAPI.h": "MaterialCAPI_8h.html",
    "../src/materials/MaterialProvider.hh": "MaterialProvider_8hh.html",
}


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

        if (line[index] == "[" and not escaped(line, index)
                and (index == 0 or line[index - 1] != "!")):
            label_end = line.find("]", index + 1)
            if label_end > index + 1 and label_end + 1 < len(line) and line[label_end + 1] == "(":
                target_end = line.find(")", label_end + 2)
                if target_end > label_end + 2:
                    target = line[label_end + 2:target_end]
                    target_path = target.split(maxsplit=1)[0]
                    path_without_fragment = target_path.split("#", 1)[0].split("?", 1)[0]
                    suffix = Path(path_without_fragment).suffix.lower()
                    artifact_marker = "build/verification/"
                    artifact_offset = path_without_fragment.find(artifact_marker)
                    is_local = (
                        not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target_path)
                        and not target_path.startswith(("/", "//"))
                        and "://" not in target_path
                    )
                    is_data_asset = suffix in {".csv", ".json", ".jsonl", ".txt", ".log"}
                    if is_local and path_without_fragment in DOXYGEN_HEADER_PAGES:
                        label = line[index + 1:label_end]
                        generated_page = DOXYGEN_HEADER_PAGES[path_without_fragment]
                        output.extend(("[", label, "](", generated_page, ")"))
                        index = target_end + 1
                        continue

                    if is_local and (artifact_offset >= 0 or is_data_asset
                                     or path_without_fragment.endswith("/")):
                        label = line[index + 1:label_end]
                        artifact = (path_without_fragment[artifact_offset:]
                                    if artifact_offset >= 0 else path_without_fragment)
                        output.append(label)
                        output.append(" (`")
                        output.append(artifact)
                        output.append("`)")
                        index = target_end + 1
                        continue

        # Doxygen does not resolve Markdown file links with an anchor to its
        # generated page. Strip only the fragment in the filtered Doxygen copy;
        # the checked-in Markdown keeps the anchor for GitHub navigation.
        if line.startswith("](", index) and not escaped(line, index):
            end = line.find(")", index + 2)
            if end > index + 2:
                destination = line[index + 2:end]
                path, marker, _ = destination.partition("#")
                if (marker and path.endswith(".md") and "://" not in path
                        and not path.startswith(("/", "//"))):
                    output.extend(("](", path, ")"))
                    index = end + 1
                    continue

        if line.startswith(r"\(", index) and not escaped(line, index):
            end = line.find(r"\)", index + 2)
            if end > index + 2:
                output.extend((r"\f$", line[index + 2:end], r"\f$"))
                index = end + 2
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
    math_delimiter: Optional[str] = None
    ordinary_fence: Optional[Tuple[str, int]] = None

    for line in text.splitlines(keepends=True):
        if ordinary_fence is not None:
            output.append(line)
            if closes_fence(line, *ordinary_fence):
                ordinary_fence = None
            continue

        if pending_math is not None:
            closes_math = (
                closes_fence(line, *math_fence)
                if math_fence is not None
                else (DISPLAY_MATH_DELIMITER.fullmatch(line) is not None
                      if math_delimiter == "$$"
                      else DISPLAY_MATH_CLOSE.fullmatch(line) is not None)
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
            math_delimiter = None
            continue

        if (DISPLAY_MATH_DELIMITER.fullmatch(line) is not None
                or DISPLAY_MATH_OPEN.fullmatch(line) is not None):
            pending_math = [line]
            math_fence = None
            math_delimiter = "$$" if DISPLAY_MATH_DELIMITER.fullmatch(line) else r"\["
            continue

        parsed = fence(line)
        if parsed is None:
            output.append(translate_inline_math(line))
            continue

        marker, length, info = parsed
        if marker == "`" and info == "math":
            pending_math = [line]
            math_fence = (marker, length)
            math_delimiter = None
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
