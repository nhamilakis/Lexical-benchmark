# Run Training Jobs

For training the models you need to make sure first the dataset configuration is correctly setup (see settings).


# Training on Cluster

Train a single model


> !!!! Before training you need to download the tokenizer on jean-zay compute nodes do not have internet access.
> ```uv run hf-download```


```bash
export CLUSTER_NAME="jean-zay" # configure jz data dir 
# See available arguments using 
uv run scripts/train/train.py single --help
# Run job using sbatch
$ sbatch code/src/slurm_scripts/training/train-jz.sh stela EN 2 0 lstm --resume
uv run code/src/scripts/train/train.py single stela EN 2 0 lstm --batch-size 256
```

Train in multiple using sbatch array

```bash
export CLUSTER_NAME="jean-zay" # configure jz data dir
# See available options
uv run scripts/train/training-array-args.py --help
# generate array file
uv run scripts/train/training-array-args.py --dataset-name stela --train-chunks 0,1 --split-include 1,2,3,4,5,6 --model-types gpt2 --gpt2-batch-size 128 --to-args -s
# Run the sbatch array
sbatch --array=0-12 slurm_scripts/training/train-jz.sh /path/to/to_train.toml 

sbatch --array=0 code/src/slurm_scripts/training/train-jz.sh to_train.toml 

```


> Hint: you can also use `--previex` instead of `--to-args` if you want to make sure you picked the correct filters.


For adapting the amount of ressources used see the `train-jz.sh` header and adapt it to your needs. 

## Oberon

- Use the `train-oberon.sh` file instead of the `train-jz.sh`

- The data dir discovery happens automaticly (should remove the `CLUSTER_NAME="jean-zay"`)

## Other cluster

- You need to define your custom slurm script in the training folder.

- Configure the data directory by exporting the `export DATA_DIR=/path/to/data`



