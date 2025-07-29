import typing as t

######################################
# Types to separate different strings
WordStr = str  # A string containing a single word
SentenceStr = str  # A string containing a sentence of words
TokenizedSentenceStr = str  # A string containing a sentence of words formatted by word-tokenizer
######################################

MODEL_TYPE = t.Literal["lstm", "gpt2"]
ESTIMATION_TYPE = t.Literal["100hpy", "500hpy", "1000hpy"]
DEVICE_TYPE = t.Literal["cuda", "cpu", "mps"]
TRAINABLE_DATASETS = t.Literal["stela", "childes"]
AVAILABLE_LANGS = t.Literal["EN"]
