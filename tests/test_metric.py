from lexical_benchmark.utils import hf_util
import string 
# test tokenizer 
def add_special_tokens(special_token_lst: list[str] = ["'", "|"]):
    for special_token in special_token_lst:
            tokenizer.add_tokens(special_token)
    return tokenizer
# Tokenizer setup
model_max_length=2048
chars = string.ascii_letters
tokenizer = hf_util.CharacterTokenizer(chars=chars, model_max_length=model_max_length)
tokenizer = add_special_tokens()
print("Special tokens have been added")





print(tokenizer.get_id("s"))
print(tokenizer.get_id("|"))