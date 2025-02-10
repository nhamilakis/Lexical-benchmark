jupyter_port := "9998"
compute_node := "puck1"
current_dir := justfile_directory()
COML_CLUSTER := "oberon2"
scratch1_deploy_folder := "/scratch1/projects/lexical-benchmark/v2/jean-zay-code/Lexical_benchmark"
JZ_CLUSTER := "jean-zay"
JZ_SCRATCH_WORK := "/lustre/fsn1/projects/rech/hhb/ucx81cx/work"
JZ_SRC_DEV := JZ_SCRATCH_WORK + "/dev/code"
JZ_SRC_TEST_1 := JZ_SCRATCH_WORK + "/test1/code"
JZ_SRC_TEST_2 := JZ_SCRATCH_WORK + "/test2/code"

hostname := `hostname`
COML_WORKSPACE := if hostname == "NicolasMBP.local" {
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
    rsync -azP --delete --exclude=".venv" --exclude=".ipynb_checkpoints" "{{COML_CLUSTER}}:{{COML_WORKSPACE}}/notebooks/" "{{current_dir}}/notebooks/"

[doc("Deploy source code to remote")]
deploy-oberon: exec-permissions
    echo "Syncing source-code directory..."
    rsync -azP --delete --exclude=".venv" --exclude="data" --exclude=".mypy_cache" --exclude="notebooks" --exclude=".ruff_cache" --exclude="*.egg-info" "{{current_dir}}/" "{{COML_CLUSTER}}:{{COML_WORKSPACE}}/source/"

[doc("Deploy source code to remote")]
deploy-coml-prod: exec-permissions
    echo "Syncing source-code directory..."
    rsync -azP --delete --exclude=".venv" --exclude=".mypy_cache" --exclude="notebooks" --exclude=".ruff_cache" --exclude="src/*.egg-info" "{{current_dir}}/" "{{COML_CLUSTER}}:{{scratch1_deploy_folder}}"

[doc("Deploy source code to jean-folder test folder !!")]
deploy-jz-dev: exec-permissions
    #!/bin/bash
    # SSHPASS not set
    echo "Syncing source-code directory..."
    if [ -z "${SSHPASS}" ]; then
        rsync -azP --delete --exclude=".venv" --exclude="data" --exclude=".mypy_cache" --exclude="notebooks" --exclude=".ruff_cache" --exclude="src/*.egg-info" "{{current_dir}}/" "{{JZ_CLUSTER}}:{{JZ_SRC_DEV}}"
    else
        sshpass -e rsync -azP --delete --exclude=".venv" --exclude="data" --exclude=".mypy_cache" --exclude="notebooks" --exclude=".ruff_cache" --exclude="src/*.egg-info" "{{current_dir}}/" "{{JZ_CLUSTER}}:{{JZ_SRC_DEV}}"
    fi

[doc("Deploy source code to jean-folder in test env 1")]
deploy-jz-test1: exec-permissions
    #!/bin/bash
    # SSHPASS not set
    echo "Syncing source-code directory..."
    if [ -z "${SSHPASS}" ]; then
        rsync -azP --delete --exclude=".venv" --exclude="data" --exclude=".mypy_cache" --exclude="notebooks" --exclude=".ruff_cache" --exclude="src/*.egg-info" "{{current_dir}}/" "{{JZ_CLUSTER}}:{{JZ_SRC_TEST_1}}"
    else
        sshpass -e rsync -azP --delete --exclude=".venv" --exclude="data" --exclude=".mypy_cache" --exclude="notebooks" --exclude=".ruff_cache" --exclude="src/*.egg-info" "{{current_dir}}/" "{{JZ_CLUSTER}}:{{JZ_SRC_TEST_1}}"
    fi

[doc("Deploy source code to jean-folder in test env 2")]
deploy-jz-test2: exec-permissions
    #!/bin/bash
    # SSHPASS not set
    echo "Syncing source-code directory..."
    if [ -z "${SSHPASS}" ]; then
        rsync -azP --delete --exclude=".venv" --exclude="data" --exclude=".mypy_cache" --exclude="notebooks" --exclude=".ruff_cache" --exclude="src/*.egg-info" "{{current_dir}}/" "{{JZ_CLUSTER}}:{{JZ_SRC_TEST_2}}"
    else
        sshpass -e rsync -azP --delete --exclude=".venv" --exclude="data" --exclude=".mypy_cache" --exclude="notebooks" --exclude=".ruff_cache" --exclude="src/*.egg-info" "{{current_dir}}/" "{{JZ_CLUSTER}}:{{JZ_SRC_TEST_2}}"
    fi

[doc("Connect to jean-zay server")]
connect-jz:
    #!/bin/bash
    # SSHPASS not set
    if [ -z "${SSHPASS}" ]; then
        ssh jean-zay
    else
        sshpass -e ssh jean-zay
    fi

[doc("Make executables")]
exec-permissions:
    find src/scripts -name "*.py" -exec chmod +x {} \;
    find experiments -name "*.sh" -exec chmod +x {} \;

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