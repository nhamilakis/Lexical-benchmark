jupyter_port := "9998"
compute_node := "puck1"
current_dir := justfile_directory()
COML_CLUSTER := "oberon2"
JZ_CLUSTER := "jean-zay"
scratch1_deploy_folder := "/scratch1/projects/lexical-benchmark/v2/jean-zay-code/Lexical_benchmark"
jean_zay_deploy_folder_prod := "/lustre/fswork/projects/rech/hhb/ucx81cx/code/Lexical_benchmark"
jean_zay_deploy_folder_dev := "/lustre/fsn1/projects/rech/hhb/ucx81cx/code"

hostname := `hostname`
COML_WORKSPACE := if hostname == "Nicolass-MBP.lan" {
    "workspace/src/LexicalBenchmark2"
} else if hostname == "other-person" {
    "projects/LexicalBenchmark"
} else {
    "code/LexicalBenchmark"
}


_default:
  @just --choose

[doc("Open SSH tunnel for remote notebook server.")]
notebook-tunnel node=compute_node port=jupyter_port:
    @echo "Creating a tunnel to {{node}}:{{port}}"
    ssh -L "{{port}}:{{node}}:{{port}}" "{{node}}" -N


[doc("Fetch notebooks from Oberon")]
fetch-notebooks:
    echo "Fetching notebooks..."
    rsync -azP --delete --exclude=".ipynb_checkpoints" "{{COML_CLUSTER}}:{{COML_WORKSPACE}}/notebooks/" "{{current_dir}}/notebooks/"

[doc("Deploy experiment code to remote")]
deploy-experiments:
    echo "Syncing experiment directory..."
    rsync -azP --delete --exclude=".mypy_cache" --exclude="notebooks" --exclude=".ruff_cache" --exclude="*.egg-info" "{{current_dir}}/experiments/" "{{COML_CLUSTER}}:{{COML_WORKSPACE}}/experiments/"

[doc("Deploy source code to remote")]
deploy-source: 
    echo "Syncing source-code directory..."
    rsync -azP --delete --exclude=".mypy_cache" --exclude="notebooks"  --exclude="experiments" --exclude=".ruff_cache" --exclude="*.egg-info" "{{current_dir}}/" "{{COML_CLUSTER}}:{{COML_WORKSPACE}}/source/"

[doc("Deploy source code to remote")]
deploy-source-coml-prod: 
    echo "Syncing source-code directory..."
    rsync -azP --delete --exclude=".mypy_cache" --exclude="notebooks" --exclude=".ruff_cache" --exclude="src/*.egg-info" "{{current_dir}}/" "{{COML_CLUSTER}}:{{scratch1_deploy_folder}}"

[doc("Deploy source code to jean-folder in production")]
deploy-jean-zay-prod: 
    echo "Syncing source-code directory..."
    rsync -azP --delete --exclude="data" --exclude=".mypy_cache" --exclude="notebooks" --exclude=".ruff_cache" --exclude="src/*.egg-info" "{{current_dir}}/" "{{JZ_CLUSTER}}:{{jean_zay_deploy_folder_prod}}"

[doc("Deploy source code to jean-folder in debug mode")]
deploy-jean-zay-dev: 
    echo "Syncing source-code directory..."
    rsync -azP --delete --exclude="data" --exclude=".mypy_cache" --exclude="notebooks" --exclude=".ruff_cache" --exclude="src/*.egg-info" "{{current_dir}}/" "{{JZ_CLUSTER}}:{{jean_zay_deploy_folder_dev}}"

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