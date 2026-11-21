"""Load the store's help documents and split them into section-sized chunks."""
from dataclasses import dataclass
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


@dataclass(frozen=True)
class Chunk:
    source: str   # file name, used for citations
    heading: str  # section heading
    text: str     # heading + body


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Return (heading, body) pairs. Text before the first heading is 'Overview'."""
    sections: list[tuple[str, str]] = []
    heading, buffer = "Overview", []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((heading, body))

    for line in markdown.splitlines():
        if line.startswith("#"):
            flush()
            heading, buffer = line.lstrip("#").strip(), []
        else:
            buffer.append(line)
    flush()
    return sections


def load_chunks(folder: str | Path = DOCS_DIR) -> list[Chunk]:
    chunks = []
    for path in sorted(Path(folder).glob("*.md")):
        for heading, body in split_sections(path.read_text(encoding="utf-8")):
            chunks.append(Chunk(path.name, heading, f"{heading}. {body}"))
    return chunks
