#!/bin/bash
#SBATCH --job-name=lb-generation-debug
#SBATCH --account=hhb@a100
# Partition (A100)
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
# Number of GPUs per task (On a100 8 GPUs per node are available.)
#SBATCH --gres=gpu:1
# Number of cores per task for gpu_p5 (1/8 of 8-GPUs A100 node)
# A100 nodes have 64 cores, should use proportional to GPU number (1 gpu 1/8 of the CPUs)
# For 4 GPUs use 32 cores per task
#SBATCH --cpus-per-task=8
# Only run this when testing
##SBATCH --qos=qos_gpu_a100-dev
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/fswork/projects/rech/hhb/ucx81cx/logs/generation-%j-%a.log
#SBATCH --hint=nomultithread        # hyperthreading is deactivated

# ENV setup, if not set
if [[ -z "${_LM_ENV}" ]]; then
    source $WORK/load.sh
fi

export MODEL_ROOT="$WORK/model_light"
export GEN_ROOT="$WORK/jz-gen"
export DATASET_ROOT="$WORK/datasets"
export CODE="$WORK/code"
export _LM_ENV="active"
export JZ=1

if [[ -z "${1}" ]]; then
    echo "Error: index file required" >&2
    exit 1
fi
JOB_INDEX_FILE=$1
if [[ -z "${2}" ]]; then
    echo "Error: specify line in file required" >&2
    exit 1
fi
JOB_ID=$2

get_line() {
    local file="$1"
    local n="$2"

    # Check if both arguments are provided
    if [[ $# -ne 2 ]]; then
        echo "Usage: get_line <file> <line_number>" >&2
        return 1
    fi

    # Check if file exists
    if [[ ! -f "$file" ]]; then
        echo "Error: File '$file' not found" >&2
        return 1
    fi

    # Check if n is a number
    if ! [[ "$n" =~ ^[0-9]+$ ]]; then
        echo "Error: Line number must be a non-negative integer" >&2
        return 1
    fi

    # Get the line (adding 1 because sed uses 1-based indexing)
    n=$((n + 1))
    sed "${n}q;d" "$file"
}



echo "=== SLURM Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Number of Nodes: $SLURM_JOB_NUM_NODES"
echo "CPUs per Node: $SLURM_CPUS_ON_NODE"
echo "Allocated GPUs: $SLURM_GPUS"
echo "GPU List: $CUDA_VISIBLE_DEVICES"

# Print system information
echo -e "\n=== System Information ==="
echo "Hostname: $(hostname)"
echo "CPU Info: $(lscpu | grep 'Model name' | sed 's/Model name: *//')"
echo "Memory Info: $(free -h | grep Mem)"

echo -e "\n=== PYTHON ==="
echo "python: $(which python)"
echo "python-version $(python -V)"


echo "Running Generation  ($SLURM_ARRAY_JOB_ID/$JOB_ID) @ $(date)"

# Grab parameters from index file
read model output <<< "$(get_line "${JOB_INDEX_FILE}" $2)"

python $CODE/Lexical_benchmark/src/scripts/generation/generate.py --gen_file "$WORK/oberon-gen/CHILDES_model.csv" --model_path "$MODEL_ROOT/$model" --generation_path "$GEN_ROOT/$output" --debug "True"

echo "Completed Generation  ($SLURM_ARRAY_JOB_ID/$JOB_ID) @ $(date)"
