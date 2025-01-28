
import string
import random
from lexical_benchmark.utils import hf_util
from vllm import LLM, SamplingParams

# load tokenizer
chars = string.ascii_letters
tokenizer = hf_util.CharacterTokenizer(chars=chars, model_max_length=1024)
print("Tokenizer has been loaded")

model_path = "/scratch1/projects/lexical-benchmark/v2/models/STELATranscriptions2/by_month/EN/10/00/trans"
random_token_id = random.randint(0, 25)

temp = 1.0
word_num = 1
model = LLM(model=model_path,skip_tokenizer_init = True)
print("Model has been loaded")

sampling_params = SamplingParams(
                        temperature=temp,
                        max_tokens=word_num,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                    )

tokens = tokenizer.decode([random_token_id])
print(f"Random token is {random_token_id}. Tokens are {tokens}")

outputs = model.generate(prompt_token_ids = [random_token_id],sampling_params = sampling_params)
print("Having finished generation!")
print(outputs)

if not outputs or not outputs[0].outputs:
    raise ValueError("vLLM generated empty output")
gen = list(outputs[0].outputs[0].token_ids)
print(gen)
# decode the generations
gen_tokens = tokenizer.decode(gen)
print(gen_tokens)
print(tokens+gen_tokens)
