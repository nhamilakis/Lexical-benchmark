#!/home/nhamilakis/envs/venvs/lbenchmark/bin/python3.11
# fmt: off
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=benchmark-pos
#SBATCH --time=2:00:00
#SBATCH --export=ALL
#SBATCH --output benchmark-pos-%J.log
# fmt: on
import pprint
import random
import time

import nltk
import spacy
from lexical_benchmark.datasets import utils as dataset_utils


def get_en_random_words(n: int, min_length: int = 2) -> list[str]:
    """Generate a list of n random English words."""
    # Download wordnet if not already downloaded
    nltk.download("wordnet")
    from nltk.corpus import wordnet

    # Get all lemmas (base forms of words)
    all_words = [w for w in wordnet.words(lang="eng") if w.isalpha() and len(w) >= min_length]

    return random.sample(all_words, k=n)


def benchmark_batch_size(words: list[str], nlp: spacy.Language, sizes: list[int]) -> dict[int, float]:
    """Benchmark different batch sizes."""
    timings = {}
    for size in sizes:
        start = time.perf_counter()
        dataset_utils.batch_word_to_pos(words, nlp, batch_size=size)
        timings[size] = time.perf_counter() - start
    return timings



# Run benchmark
words = get_en_random_words(2048)
pos_model = dataset_utils.various.spacy_model("en_core_web_trf", require_gpu=True)
sizes = [32, 64, 128, 256, 512, 1024, 2048]
timings = benchmark_batch_size(words, pos_model, sizes)

pprint.pprint(timings)
