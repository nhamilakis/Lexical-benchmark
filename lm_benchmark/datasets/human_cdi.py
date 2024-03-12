"""
This is a loader and extractor for Human CDI from wordbank

https://wordbank.stanford.edu/data?name=item_data

More info: docs/datasets/human_wordbank

"""


class GoldReference:
    """
    Downloaded from website wordbank (CSV Format)

    Extract:

    Month columns (XX - XX) -> after category and before word (in the CSV)
    Proportion of children that knows the word

    Word column (item_definition -> requires cleaning)

    POS (Part of speech) -> derived from Word column (using word2POS function)

    Word Length (derived from Word count number of chars after cleaning)
    """
    pass


class CHILDES:
    """
        Use Jing's version of CHILDES (on oberon) extract from annotation CHI,ADULT(all) speech
        clean up all things that are not words.

        Create  a CSV file containing a list of utterances


        Output:

        - child, speaker, language, corpus, number_of_tokens, src_path, filename, content
        - Word Frequency table
    """
    pass



