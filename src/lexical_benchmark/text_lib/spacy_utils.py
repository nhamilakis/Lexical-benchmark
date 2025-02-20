import collections
import typing as t

if t.TYPE_CHECKING:
    try:
        from spacy import Language
    except ImportError:
        Language = t.Any


def spacy_model(model_name: str, *, require_gpu: bool = True) -> "Language":  # type: ignore[private-import-usage]
    """Safely load spacy Language Model."""
    if require_gpu:
        import spacy

        spacy.require_gpu()
    else:
        import spacy

        spacy.prefer_gpu()

    try:
        return spacy.load(model_name)
    except OSError:
        from spacy.cli.download import download

        download(model_name)

        return spacy.load(model_name)


def phrase_to_pos(phrase: str, pos_model: "Language") -> dict[str, list[str]]:
    """Extract POS tags for all tokens in a given phrase."""
    pos_mapping = collections.defaultdict(list)
    doc = pos_model(phrase)
    for token in doc:
        pos_mapping[token.text].append(token.pos_)
    return dict(pos_mapping)


def batch_phrase_list_to_pos(
    phrases: list[str], pos_model: "Language", n_process: int = 1, batch_size: int = 32
) -> dict[str, list[str]]:
    """Infer Part of Speech from a given list of phrases."""
    docs = pos_model.pipe(phrases, batch_size=batch_size, n_process=n_process)
    pos_mapping = collections.defaultdict(list)
    for doc in docs:
        for token in doc:
            pos_mapping[token.text].append(token.pos_)
    return dict(pos_mapping)
