#!/lustre/fsn1/projects/rech/hhb/ucx81cx/conda/lm_dev/bin/python
# fmt: off
#SBATCH --job-name=test-batch-size
#SBATCH --account=hhb@a100
#SBATCH -C a100                 # Partition (A100)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Number of GPUs per task (On a100 8 GPUs per node are available.)
#SBATCH --gres=gpu:1
# Number of cores per task for gpu_p5 (1/8 of 8-GPUs A100 node)
# A100 nodes have 64 cores, should use proportional to GPU number (1 gpu 1/8 of the CPUs)
# For 4 GPUs use 32 cores per task
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=%x-%j-%a.log
#SBATCH --hint=nomultithread    # hyperthreading is deactivated
# Only run this when testing
#SBATCH --qos=qos_gpu_a100-dev
# fmt: on
import pprint
import random
import time

import spacy

from lexical_benchmark.datasets import utils as dataset_utils


def get_en_random_words(n: int, min_length: int = 2) -> list[str]:
    """Generate a list of n random English words."""
    # Download wordnet if not already downloaded
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
print("Computing words")

words = get_en_random_words(2048)
print(f"Made {len(words)} !")
print("Loading model : en_core_web_trf on gpu")
pos_model = dataset_utils.various.spacy_model("en_core_web_trf", require_gpu=True)
sizes = [32, 64, 128, 256, 512, 1024, 2048]
print("Running batching test looo")
timings = benchmark_batch_size(words, pos_model, sizes)
print("Testing completed, results: ")
pprint.pprint(timings)
