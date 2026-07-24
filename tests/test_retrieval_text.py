import pytest

from localml_scholar.retrieval import (
    LexicalTokenizerConfig,
    lexical_terms,
    tokenize_lexically,
)


def test_lexical_policy_handles_words_code_numbers_and_unicode() -> None:
    text = "Scaled Dot-Product can't alter learning_rate or CamelCase42; λ=3.14 café."

    terms = lexical_terms(text)

    assert [term.term for term in terms] == [
        "scaled",
        "dot",
        "product",
        "can't",
        "alter",
        "learning_rate",
        "or",
        "camelcase42",
        "λ",
        "3.14",
        "café",
    ]
    assert all(
        text[term.start_character : term.end_character] == term.original
        for term in terms
    )
    assert [term.position for term in terms] == list(range(len(terms)))


def test_casefolding_is_explicit_and_source_spans_remain_original() -> None:
    folded = lexical_terms("Straße STRASSE")
    exact = tokenize_lexically(
        "Straße STRASSE",
        LexicalTokenizerConfig(casefold=False),
    )

    assert [term.term for term in folded] == ["strasse", "strasse"]
    assert [term.original for term in folded] == ["Straße", "STRASSE"]
    assert exact == ("Straße", "STRASSE")


def test_repeated_terms_and_whitespace_are_deterministic() -> None:
    assert tokenize_lexically(" term\tTERM\nterm ") == ("term", "term", "term")
    assert tokenize_lexically("") == ()
    assert tokenize_lexically(" \n\t ") == ()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"normalization": "nfc"},
        {"split_hyphens": False},
        {"preserve_apostrophes": False},
        {"preserve_underscores": False},
        {"split_camel_case": True},
    ],
)
def test_unsupported_lexical_policies_fail_explicitly(kwargs: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        LexicalTokenizerConfig(**kwargs)
