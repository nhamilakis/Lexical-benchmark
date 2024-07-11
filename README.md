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

TBA


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




