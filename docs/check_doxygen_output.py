#!/usr/bin/env python3
"""Smoke-check the generated solver documentation and local diagram assets."""

from argparse import ArgumentParser
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
from typing import List, Optional, Tuple
from urllib.parse import unquote, urlsplit


class LocalReferences(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.references: List[str] = []

    def handle_starttag(
        self, tag: str, attributes: List[Tuple[str, Optional[str]]]
    ) -> None:
        if tag not in {"a", "img", "object"}:
            return
        attribute = {name: value for name, value in attributes}
        key = "href" if tag == "a" else "src" if tag == "img" else "data"
        if attribute.get(key):
            self.references.append(attribute[key] or "")


def local_target(html_file: Path, reference: str) -> Optional[Path]:
    parsed = urlsplit(reference)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    return (html_file.parent / unquote(parsed.path)).resolve()


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("html_directory", type=Path)
    parser.add_argument("--expect-diagrams", action="store_true")
    arguments = parser.parse_args()

    html_directory = arguments.html_directory.resolve()
    overview = html_directory / "solver_algorithms.html"
    errors: List[str] = []
    if not overview.is_file():
        errors.append(f"missing solver overview: {overview}")
    else:
        references = LocalReferences()
        references.feed(overview.read_text(encoding="utf-8"))
        for reference in references.references:
            target = local_target(overview, reference)
            if target is not None and not target.exists():
                errors.append(f"broken overview reference: {reference}")

    rendered_pages = [
        path for path in html_directory.glob("*.html")
        if not path.name.endswith("_source.html")
    ]
    rendered_text = "\n".join(path.read_text(encoding="utf-8") for path in rendered_pages)
    if re.search(r"(?:@|\\)(?:startuml|enduml)\b", rendered_text):
        errors.append("raw PlantUML commands leaked into rendered HTML")

    diagram_references = sorted(set(re.findall(
        r"inline_umlgraph_[^\"'<> ]+\.(?:svg|png)", rendered_text)))
    for reference in diagram_references:
        asset = html_directory / reference
        if not asset.is_file():
            errors.append(f"missing diagram asset: {reference}")
        elif reference.endswith(".svg"):
            svg = asset.read_text(encoding="utf-8")
            if "Syntax Error" in svg or "<svg" not in svg:
                errors.append(f"PlantUML reported an invalid activity diagram: {reference}")
    if arguments.expect_diagrams:
        if len(diagram_references) < 8:
            errors.append(
                f"expected at least 8 rendered solver diagrams, found {len(diagram_references)}")
        raster = [reference for reference in diagram_references if reference.endswith(".png")]
        if raster:
            errors.append("diagram-enabled documentation produced raster PlantUML assets: " + ", ".join(raster))

    if errors:
        print("Doxygen output smoke check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(f"Doxygen output smoke check passed ({len(diagram_references)} diagram assets).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
