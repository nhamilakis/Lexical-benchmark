# All files created belong to the project
umask 007

srun --pty --job-name="interactive" --cpus-per-task="4" -t "4:00:00" --partition=gpu bash -i