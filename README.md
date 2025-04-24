# Machine CDI: Lexical-benchmark for language acquisition


## Getting started


#### Installation

To get started with this module you will need to have the following requirements : [uv](https://docs.astral.sh/uv/getting-started/installation/) 


Recommended way use the installation script : 

```
curl -LsSf https://raw.githubusercontent.com/nhamilakis/Lexical-benchmark/refs/heads/dev/install.sh | sh
```

This script does all the setup required & creates a local env with all requirements.


### Using Module

This module is mostly a library allowing to be imported to perform the various tasks and create analytics from the datasets.
There are two different mediums

A list of scripts in the [src/scripts/](src/scripts/) folder allow to do most of the computations

TBA...


A list of notebooks allowing to explore data and plot analytics.


TBA...


> If you are working in a slurm cluster all scripts used to run the experiments can be found @ [src/slurm_scripts/](src/slurm_scripts/)


## Datasets

For more detailed information on each dataset you can check out their pages :

- [CHILDES/EN](docs/datasets/childes.md)

- [Wordbank-CDI/EN](docs/datasets/wordbank-cdi.md)

- [STELLA/EN](docs/datasets/stella.md)

- [ChildRealistic](docs/datasets/child-realistic.md) (TBA)

- Details on how the word-lists used for word validation -> [wordlist](docs/wordlist.md)