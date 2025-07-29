#!/bin/bash
#SBATCH --job-name=backup
#SBATCH --account=hhb@cpu
#SBATCH --partition=archive
#SBATCH --nodes=1
#SBATCH --mail-type=BEGIN,END,FAIL  # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=nh@cognitive-ml.fr
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
# Only run this when testing
#SBATCH --time=18:00:00
echo "---START OF ARCHIVE SCRIPT--- $(date)"

# All files created belong to the project
umask 007

# archive_folder.sh - Script to archive a folder with date in filename
# Usage: ./archive_folder.sh [source_folder]
DEST_DIR="/lustre/fsstor/projects/rech/hhb/commun"

# Check if correct number of arguments are provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 [source_folder]"
    exit 1
fi

# Get source folder path (remove trailing slash if present)
SOURCE_FOLDER=${1%/}

# Check if source folder exists
if [ ! -d "$SOURCE_FOLDER" ]; then
    echo "Error: Source folder '$SOURCE_FOLDER' does not exist."
    exit 1
fi

FOLDER_NAME=$(basename "$SOURCE_FOLDER")
CURRENT_DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
ARCHIVE_FILE="$DEST_DIR/${FOLDER_NAME}_${CURRENT_DATETIME}.tar.gz"

echo "Archiving '$SOURCE_FOLDER' to '$ARCHIVE_FILE'..."
tar -czf "$ARCHIVE_FILE" -C "$(dirname "$SOURCE_FOLDER")" "$(basename "$SOURCE_FOLDER")"

# Check if archiving was successful
if [ $? -eq 0 ]; then
    echo "Archive created successfully: $ARCHIVE_FILE"
    echo "Archive size: $(du -h "$ARCHIVE_FILE" | cut -f1)"
else
    echo "Error: Failed to create archive."
    exit 1
fi

echo "---END OF ARCHIVE SCRIPT--- $(date)"
exit 0
