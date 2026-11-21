from support_bot.knowledge import load_chunks, split_sections


def test_split_sections_by_heading():
    md = "# Title\n\n## One\nfirst body\n\n## Two\nsecond body\n"
    assert split_sections(md) == [("One", "first body"), ("Two", "second body")]


def test_load_chunks_reads_all_docs():
    chunks = load_chunks()
    assert len(chunks) >= 15
    assert all(c.source.endswith(".md") for c in chunks)
