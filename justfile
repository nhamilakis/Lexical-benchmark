jupyter_port := "9998"
compute_node := "puck1"
current_dir := justfile_directory()
remote := "oberon2"
remote_notebook_path := "workspace/src/LexicalBenchmark2/notebooks/"
remote_source_path := "workspace/src/LexicalBenchmark2/source/"

_default:
  @just --choose

[doc("Open SSH tunnel for remote notebook server.")]
notebook-tunnel node=compute_node port=jupyter_port:
    @echo "Creating a tunnel to {{node}}:{{port}}"
    ssh -L "{{port}}:{{node}}:{{port}}" "{{node}}" -N -v -v

[doc("Fetch notebooks from Oberon")]
fetch-notebooks:
    echo "Fetching notebooks..."
    rsync -azP --delete --exclude=".ipynb_checkpoints" "{{remote}}:{{remote_notebook_path}}" "{{current_dir}}/notebooks/"

[doc("Deploy source code to Oberon")]
deploy:
    echo "Syncing current directory..."
    rsync -azP --delete --exclude=".mypy_cache" --exclude="notebooks" --exclude=".ruff_cache" --exclude="*.egg-info" "{{current_dir}}/" "{{remote}}:{{remote_source_path}}"

[doc("Install module & dependencies")]
install:
    pip install -e ".[dev]"
    mypy --install-types

[doc("Run Jupyter Server Locally")]
run-notebook:
    jupyter lab

[doc("Check Syntax (RUFF)")]
syntax-check:
    ruff check

[doc("Check Typing (mypy)")]
type-check:
    mypy lexical_benchmark

[doc("Auto Formatting (RUFF)")]
format:
    ruff format lexical_benchmark

[doc("Commit and push all changes")]
add-commit-push m="":
    # git add .
    @[[ ! -z "{{m}}" ]] &&  echo "commiting:: {{m}}" # git commit -m "{{m}}"
    @[[ -z "{{m}}" ]] &&  echo "commiting:: empty" # git commit -m "{{m}}"
    # git push
