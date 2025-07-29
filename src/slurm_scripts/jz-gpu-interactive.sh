#!/bin/bash

# All files created belong to the project
umask 007

# Initialize variables
PROJECT="$IDRPROJ"
time_value="02:00:00"
nproc_value="8"
ngpu_value="1"
gpu_partition="h100"

# Function to display help
show_help() {
    echo "Usage: $0 [-t time] [-p nproc] [-g ngpu] [-h]"
    echo "Options:"
    echo "  -t <time>    Specify the time (default: 2h)"
    echo "  -p <nproc>   Specify the number of processors (default: 8)"
    echo "  -g <ngpu>    Specify the number of GPUs (default: 1)"
    echo "  -a <a100 / h100 / v100> Specify gpu partition (default: h100)"
    echo "  -h           Show this help message"
}


# Parse command line options
while getopts "t:p:g:a:h" opt; do
    case $opt in
        t)
            time_value="$OPTARG"
            ;;
        p)
            nproc_value="$OPTARG"
            # Validate that nproc is a positive integer
            if ! [[ "$nproc_value" =~ ^[0-9]+$ ]]; then
                echo "Error: nproc must be a positive integer" >&2
                exit 1
            fi
            ;;
        g)
            ngpu_value="$OPTARG"
            # Validate that ngpu is a positive integer
            if ! [[ "$ngpu_value" =~ ^[0-9]+$ ]]; then
                echo "Error: ngpu must be a positive integer" >&2
                exit 1
            fi
            ;;
        a)
            gpu_partition="$OPTARG"
            ;;
        h)
            show_help
            exit 0
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            show_help
            exit 1
            ;;
    esac
done

if [[ "$gpu_partition" == "h100" ]]; then
    echo "Runing interactive job: @H100 with CPU:$nproc_value GPU:$ngpu_value for Time:$time_value : "
    srun --pty --job-name="interactive-gpu" --account="$PROJECT@h100" --nodes="1" --ntasks-per-node="1" --gres="gpu:$ngpu_value" --cpus-per-task="$nproc_value" -C "h100"  -t "$time_value" bash -i
elif [[ "$gpu_partition" == "a100" ]]; then
    echo "Runing interactive job: @A100 with CPU:$nproc_value GPU:$ngpu_value for Time:$time_value : "
    srun --pty --job-name="interactive-gpu" --account="$PROJECT@a100" --nodes="1" --ntasks-per-node="1" --gres="gpu:$ngpu_value" --cpus-per-task="$nproc_value" -C "a100"  -t "$time_value" bash -i
elif [[ "$gpu_partition" == "v100" ]]; then
    echo "Runing interactive job: @V100 with CPU:$nproc_value GPU:$ngpu_value for Time:$time_value : "
    srun --pty --job-name="interactive-gpu" --account="$PROJECT@v100" --nodes="1" --ntasks-per-node="1" --gres="gpu:$ngpu_value" --cpus-per-task="$nproc_value" -C "v100"  -t "$time_value" bash -i
else
    echo "Bad GPU partition specified, should be '-a [a100 | h100 | v100]'" >&2
    exit 1
fi