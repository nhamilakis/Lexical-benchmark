import spacy


def spacy_model(model_name: str) -> spacy.Language:
    """ Safely load spacy Language Model """
    try:
        return spacy.load(model_name)
    except OSError:
        from spacy.cli.download import download
        download(model_name)

        return spacy.load(model_name)
