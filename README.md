# Machine CDI: Lexical-benchmark for language acquisition

This repository focuses on a detailed comparison between human(CDI) and model performance at the lexical level, including: 

1) Receptive Vocabulary using a spot-the-word task (adapted from [BabySLM](https://github.com/MarvinLvn/BabySLM)

2) Expressive Vocabulary using (un)prompted generations


## Getting started


#### Installation

To get started with this module you will need to have 


- a compatible version of python (`python3.10+`).
- The enchant library
- phonemizer

You can install the module using the following commands :

```bash
git clone https://github.com/Jing-L97/Lexical-benchmark.git
cd Lexical-Benchmark
pip install .
```

or 

```bash
git clone https://github.com/Jing-L97/Lexical-benchmark.git
cd Lexical-Benchmark
pip install -e .
```

For an editable installation (useful during devellopement).


You can also install directly from the git repository (without cloning) using:

```bash
pip install git+https://github.com/Jing-L97/Lexical-benchmark.git
```

#### Available Commands

**adjust-count** : Convert word count into accumulated monthly count.

```bash
❯ adjust-count --help
usage: adjust-count [-h] [--gen_file GEN_FILE] [--est_file EST_FILE] [--CDI_path CDI_PATH]
                    [--freq_path FREQ_PATH] [--prompt_type PROMPT_TYPE] [--lang LANG]
                    [--set SET] [--header_lst HEADER_LST] [--count COUNT]

options:
  -h, --help            show this help message and exit
  --gen_file GEN_FILE
  --est_file EST_FILE
  --CDI_path CDI_PATH
  --freq_path FREQ_PATH
  --prompt_type PROMPT_TYPE
  --lang LANG
  --set SET
  --header_lst HEADER_LST
  --count COUNT
```


**get-frequencies** : 

```bash
❯ get-frequencies --help
usage: get-frequencies [-h] [--src_file SRC_FILE] [--target_file TARGET_FILE]
                       [--header HEADER] [--ngram NGRAM]

options:
  -h, --help            show this help message and exit
  --src_file SRC_FILE
  --target_file TARGET_FILE
  --header HEADER
  --ngram NGRAM
```

**match-frequencies** :

```bash
usage: match-frequencies [-h] [--CDI_path CDI_PATH] [--human_freq HUMAN_FREQ]
                         [--machine_freq MACHINE_FREQ] [--lang LANG]
                         [--test_type TEST_TYPE] [--sampling_ratio SAMPLING_RATIO]
                         [--nbins NBINS]

options:
  -h, --help            show this help message and exit
  --CDI_path CDI_PATH
  --human_freq HUMAN_FREQ
  --machine_freq MACHINE_FREQ
  --lang LANG
  --test_type TEST_TYPE
  --sampling_ratio SAMPLING_RATIO
  --nbins NBINS
  ```

**dataset-explore** :

```bash
❯ dataset-explore --help
usage: dataset-explore [-h] [--header HEADER] [--model MODEL] [--prompt PROMPT]

options:
  -h, --help       show this help message and exit
  --header HEADER
  --model MODEL
  --prompt PROMPT
```


## Brief description

You'll probably want to start from there:
- [How to select the test set](docs/select_test.md)
- [How to run the Accumulator model](docs/accum.md)
- [How to evaluate the receptive vocabulary](docs/recep.md)
- [How to evaluate the exp vocabulary](docs/exp.md)

# Result visualization
- [How to plot the developmental curves](docs/plot_fig.md)
- [How to plot the extrapolation](docs/plot_extra.md)


# folder structure
src_data: grabbed from the internet
            
processed data: csv
    -human
    -machine
        loc
        generated data




