import re
import string
import spacy

nlp = spacy.load('en_core_web_sm')


def word_cleaning(word: str) -> str:
    """ Clean-up a word from all unnecessary characters """
    translator = str.maketrans('', '', string.punctuation + string.digits)
    clean_string = word.translate(translator).lower()
    # remove annotations; problem: polysemies
    cleaned_word = re.sub(r"\([a-z]+\)", "", clean_string)
    return cleaned_word


def word2POS(selected_words, word_type):
    """
    TODO modify IO based on requirements

    select words based on POS
    input: dataframe with column [words]
    return dataframe with selected words adn POS tags
     """
    # select open class words
    pos_all = []
    for word in selected_words['word']:
        doc = nlp(word)
        pos_lst = []
        for token in doc:
            pos_lst.append(token.pos_)
        pos_all.append(pos_lst[0])
    selected_words['POS'] = pos_all

    content_POS = ['ADJ', 'NOUN', 'VERB', 'ADV', 'PROPN']
    if word_type == 'all':
        selected_words = selected_words
    elif word_type == 'content':
        selected_words = selected_words[selected_words['POS'].isin(content_POS)]
    elif word_type == 'function':
        selected_words = selected_words[~selected_words['POS'].isin(content_POS)]

    return selected_words

