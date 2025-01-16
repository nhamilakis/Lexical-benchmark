jupyter_port := "9998"
compute_node := "puck1"
current_dir := justfile_directory()
remote := "oberon2"
remote2 := "jean-zay"
remote_asr_path := "workspace/src/LexicalBenchmark2/data-v2/asr-test/"
remote_notebook_path := "workspace/src/LexicalBenchmark2/notebooks/"
remote_experiment_path := "workspace/src/LexicalBenchmark2/experiments/"
remote_source_path := "workspace/src/LexicalBenchmark2/source/"
scratch1_deploy_folder := "/scratch1/projects/lexical-benchmark/v2/jean-zay-code/Lexical_benchmark"
jean_zay_deploy_folder := "/lustre/fswork/projects/rech/hhb/ucx81cx/code/Lexical_benchmark"

_default:
  @just --choose

[doc("Open SSH tunnel for remote notebook server.")]
notebook-tunnel node=compute_node port=jupyter_port:
    @echo "Creating a tunnel to {{node}}:{{port}}"
    ssh -L "{{port}}:{{node}}:{{port}}" "{{node}}" -N


[doc("Fetch notebooks from Oberon")]
fetch-notebooks:
    echo "Fetching notebooks..."
    rsync -azP --delete --exclude=".ipynb_checkpoints" "{{remote}}:{{remote_notebook_path}}" "{{current_dir}}/notebooks/"

fetch-asr:
    echo "Fetching asr results..."
    rsync -azP --delete "{{remote}}:{{remote_asr_path}}" "{{current_dir}}/data/asr"

[doc("Deploy experiment code to remote")]
deploy-experiments:
    echo "Syncing experiment directory..."
    rsync -azP --delete --exclude=".mypy_cache" --exclude="notebooks"  --exclude="experiments" --exclude=".ruff_cache" --exclude="*.egg-info" "{{current_dir}}/experiments/" "{{remote}}:{{remote_experiment_path}}"

[doc("Deploy source code to remote")]
deploy-source: 
    echo "Syncing source-code directory..."
    rsync -azP --delete --exclude=".mypy_cache" --exclude="notebooks"  --exclude="experiments" --exclude=".ruff_cache" --exclude="*.egg-info" "{{current_dir}}/" "{{remote}}:{{remote_source_path}}"

[doc("Deploy source code to remote")]
deploy-scratch1: 
    echo "Syncing source-code directory..."
    rsync -azP --delete --exclude=".mypy_cache" --exclude="notebooks" --exclude=".ruff_cache" --exclude="src/*.egg-info" "{{current_dir}}/" "{{remote}}:{{scratch1_deploy_folder}}"

[doc("Deploy source code to remote")]
deploy-jean-zay: 
    echo "Syncing source-code directory..."
    rsync -azP --delete --exclude="data" --exclude=".mypy_cache" --exclude="notebooks" --exclude=".ruff_cache" --exclude="src/*.egg-info" "{{current_dir}}/" "{{remote2}}:{{jean_zay_deploy_folder}}"

[doc("Deploying all elements to remote")]
deploy: deploy-source deploy-experiments

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

check-todo:
    @rg \
    --glob !notebooks/ \
    --glob !justfile \
    --glob !pyproject.toml \
    --ignore-case \
    'fixme|todo|feat' \
    .