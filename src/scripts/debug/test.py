from turtle import done


export CLUSTER_NAME="jean-zay"

src/slurm_scripts/archive-jz.sh

interactive mode
1.lstm model 
uv run code/src/scripts/train/train.py single stela EN 01 00 lstm --batch-size 256   done



2.transformer_training
uv run code/src/scripts/train/train.py single stela EN 01 00 gpt2 --batch-size 64


uv run code/src/scripts/train/training-array-args.py --dataset-name stela --train-chunks 0 --split-include 2,3,4,5,6 --model-types lstm,gpt2 --lstm-batch-size 256 --gpt2-batch-size 64  --to-args


sbatch --array=0-9 code/src/slurm_scripts/training/train-jz.sh logs/train/args/to_train.toml 


