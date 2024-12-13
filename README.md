# Machine CDI: Lexical-benchmark for language acquisition


## Getting started


#### Installation

To get started with this module you will need to have 

- a compatible version of python (`python3.10+`).

You can install the module using the following commands :

```bash
git clone -b dev https://github.com/nhamilakis/Lexical-benchmark.git
cd Lexical-Benchmark
pip install .
```

or 

```bash
git clone -b dev https://github.com/nhamilakis/Lexical-benchmark.git
cd Lexical-Benchmark
pip install -e .
```

For an editable installation (useful during devellopement).


You can also install directly from the git repository (without cloning) using:

```bash
pip install git+https://github.com/nhamilakis/Lexical-benchmark.git@dev
```

### Using Module

This module is mostly a library allowing to be imported to perform the various tasks and create analytics from the datasets.
There are two different mediums

A list of scripts in the [experiments/](experiments/) folder allow to do most of the computations

- [experiments/dataset_clean/](experiments/dataset_clean/): contains scripts allowing to setup & clean all the datasets
    - [experiments/dataset_clean/stela.py](experiments/dataset_clean/stela.py): slurm job for cleaning & formatting the STELA dataset.
    - [experiments/dataset_clean/childes.py](experiments/dataset_clean/childes.py): slurm job for cleaning & formatting the CHILDES dataset.
    - [experiments/dataset_clean/cdi.py](experiments/dataset_clean/childes.py): slurm job for cleaning & formatting the WordBank-CDI dataset.

- [experiments/dataset_clean/all_stella_duration.py](experiments/all_stella_duration.py): slurm job for measuring audio duration of stela subsets.
- [experiments/dataset_clean/asr_audio.py](experiments/asr_audio.py): slurm job for transcribing audio files using whisper.
- [experiments/dataset_clean/prepare_word_freq_maps.py](experiments/prepare_word_freq_maps.py): slurm job for preparing LexicalBenchmark data.

A list of notebooks allowing to explore data and plot analytics.


TBA...

## Datasets

For more detailed information on each dataset you can check out their pages :

- [CHILDES/EN](docs/datasets/childes.md)

- [Wordbank-CDI/EN](docs/datasets/wordbank-cdi.md)

- [STELLA/EN](docs/datasets/stella.md)

- Details on how the word-lists used for word validation -> [wordlist](docs/wordlist.md)