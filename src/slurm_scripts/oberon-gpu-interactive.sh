# All files created belong to the project
umask 007

srun --pty --job-name="interactive" --gres="gpu:1" --cpus-per-task="8" -t "4:00:00" --partition=gpu bash -i