# Run Generation Jobs

For generating text using trained the models you need to make sure first the dataset configuration is correctly setup (see settings).


# Generation on Cluster

Generate from single model


> !!!! Before any operation you need to download the tokenizer on jean-zay compute nodes do not have internet access.
> ```uv run hf-download```


```bash
export CLUSTER_NAME="jean-zay" # configure jz data dir 
# See available arguments using 
uv run scripts/train/generate.py single --help
# Run job using sbatch
$ sbatch slurm_scripts/generation/generate-jz.sh stela EN 1 0 lstm  --temperature-list 0.3,0.6 --hour-per-year "100hpy","500hpy" --resume
```

Generation in multiple using sbatch array

```bash
export CLUSTER_NAME="jean-zay" # configure jz data dir
# See available options
uv run scripts/train/generation-array-args.py --help
# generate array file
uv run scripts/train/generation-array-args.py --dataset-name stela --train-chunks 0,1 --split-include 1,2,3,4,5,6 --model-types gpt2 --temperature-list 0.3,0.4 --hour-per-year-estimations "100hpy","500hpy" --to-args -s
# Run the sbatch array
sbatch --array=0-12 slurm_scripts/generation/generate-jz.sh /path/to/to_generate.toml 
```


> Hint: you can also use `--previex` instead of `--to-args` if you want to make sure you picked the correct filters.


For adapting the amount of ressources used see the `generate-jz.sh` header and adapt it to your needs. 

## Oberon

- Use the `generate-oberon.sh` file instead of the `generate-jz.sh`

- The data dir discovery happens automaticly (should remove the `CLUSTER_NAME="jean-zay"`)

## Other cluster

- You need to define your custom slurm script in the training folder.

- Configure the data directory by exporting the `export DATA_DIR=/path/to/data`