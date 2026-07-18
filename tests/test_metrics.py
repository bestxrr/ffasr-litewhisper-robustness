from src.utils.metrics import align_words, edit_stats


def test_edit_stats_counts_ops():
    stats = edit_stats("the quick brown fox", "the brown dog")
    assert stats.substitutions == 1
    assert stats.deletions == 1
    assert stats.insertions == 0
    assert stats.ref_words == 4


def test_edit_stats_early_insertion():
    stats = edit_stats("a b c", "x a b c")
    assert stats.substitutions == 0
    assert stats.deletions == 0
    assert stats.insertions == 1
    ops = align_words("a b c", "x a b c")
    assert ops[0].kind == "insertion"
    assert ops[0].hyp == "x"


def test_edit_stats_early_deletion():
    stats = edit_stats("x a b c", "a b c")
    assert stats.substitutions == 0
    assert stats.deletions == 1
    assert stats.insertions == 0
    ops = align_words("x a b c", "a b c")
    assert ops[0].kind == "deletion"
    assert ops[0].ref == "x"
