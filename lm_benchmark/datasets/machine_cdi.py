

class LibriSpeechTrainSet:
    """
    Input:
        - Transcription files from librispeech.
        - CELEX (as CSV)

    Extract:

    CSV:
    src_file, word, word_frequency

    """
    pass


class StellaGeneration:
    """
    Input: Childes class Output

    Output:

    - CSV : childes columns + generated utterances sorted by temperature
    -
    """
    pass


class MachineCDI:
    """
    Input: LibrispeechTrainSet + Childes.word_frequency()

    Output:

        CSV = filtered Librispeech based on CHILDES content
    """
    pass



