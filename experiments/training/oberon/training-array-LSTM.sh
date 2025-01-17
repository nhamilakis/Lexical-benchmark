#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00               # Time limit hrs:min:sec
#SBATCH --output=%x-%j.log            # Standard output and error log
#SBATCH --array=0-3


ValPath="/scratch1/projects/lexical-benchmark/v2/datasets/STELATranscriptions2/dev/EN/char_hf.txt"
FILENAME="LSTM.train"

getline_split() {
    if [ $# -ne 2 ]; then
        echo "Usage: getline_split <file> <line_number>"
        return 1
    fi

    file="$1"
    n="$2"

    if [ ! -f "$file" ]; then
        echo "Error: File '$file' not found"
        return 1
    fi  # Changed this closing brace from } to fi

    # Read the line and split by comma into global variables
    IFS=',' read -r MODEL_ROOT GEN_ROOT <<< $(sed -n "$((n+1))p" "$file")

    echo "MODEL_ROOT: ${MODEL_ROOT}"
    echo "GEN_ROOT: ${GEN_ROOT}"
}




getline_split $FILENAME $SLURM_ARRAY_TASK_ID



python train_LSTM.py --TrainPath $MODEL_ROOT/char_hf.txt \
    --OutPath $GEN_ROOT \
    --ValPath $ValPath