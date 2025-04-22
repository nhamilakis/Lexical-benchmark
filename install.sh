
CURRENT_DIR=$(pwd)
GITHUB_REPO="https://github.com/nhamilakis/Lexical-benchmark.git"
if [ -d "$CURRENT_DIR/code" ]; then
   if [ -f "$CURRENT_DIR/code/pyproject.toml" ] && [ -d "$CURRENT_DIR/code/.git" ]; then
        echo "All required files and directories found!"
    else
        # Warning message showing what's missing
        echo "WARNING: The 'code' directory exists but is missing required files:"
        echo "Removing incomplete directory and re-cloning repository..."
        rm -rf "$CURRENT_DIR/code"
        git clone "$GITHUB_REPO" "$CURRENT_DIR/code"
    fi
else
    echo "Code directory does not exist. Cloning repository..."
    git clone "$GITHUB_REPO" "$CURRENT_DIR/code"
fi

mkdir -p "$CURRENT_DIR/logs"
cp -f code/deploy-dir/pyproject.toml . 
cp -f code/deploy-dir/uv.lock .
# Required two-step building to fetch flash-attn requirements
uv sync --extra build
uv sync --extra accelerate

echo "Package setup successfully!"