#!/bin/bash

# Initialize variables
time_value="2:00:00"
nproc_value="8"

# Function to display help
show_help() {
    echo "Usage: $0 [-t time] [-p nproc] [-g ngpu] [-h]"
    echo "Options:"
    echo "  -t <time>    Specify the time (default 2H)"
    echo "  -p <nproc>   Specify the number of processors (default 8)"
    echo "  -h           Show this help message"
}


# Parse command line options
while getopts "t:p:g:h" opt; do
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

echo "Runing interactive job: @CPU_P1 with CPU:$nproc_value for Time:$time_value : "
srun --pty --job-name="interactive-cpu" --account="hhb@cpu" --nodes="1" --ntasks-per-node="1" --cpus-per-task="$nproc_value" --partition=cpu_p1  -t "$time_value" bash -i