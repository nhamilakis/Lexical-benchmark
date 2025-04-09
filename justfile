jupyter_port := "9998"
compute_node := "puck1"
current_dir := justfile_directory()
COML_CLUSTER := "oberon2"
JZ_CLUSTER := "jean-zay"
JZ_SCRATCH_WORK := if hostname == "NicolasMBP.local" {
    "/lustre/fsn1/projects/rech/hhb/ucx81cx/workspace/lm_benchmark"
} else if hostname == "MBP-de-jliu" {
    "/lustre/fsn1/projects/rech/hhb/uye44va/workspace/lm_benchmark"
} else {
    "workspace/lm_benchmark"
}

JZ_SRC_DEV := JZ_SCRATCH_WORK + "/dev/code"
JZ_SRC_PROD := JZ_SCRATCH_WORK + "/prod/code"

hostname := `hostname`
COML_WORKSPACE := if hostname == "NicolasMBP.local" {
    "workspace/src/LexicalBenchmark2"
} else if hostname == "MBP-de-jliu" {
    "/home/jliu/projects/LexicalBenchmark"
} else {
    "/home/jliu/projects/LexicalBenchmark"
}

_default:
  @just --choose

[doc("Open SSH tunnel for remote notebook server.")]
notebook-tunnel node=compute_node port=jupyter_port:
    @echo "Creating a tunnel to {{node}}:{{port}}"
    ssh -L "{{port}}:{{node}}:{{port}}" "{{node}}" -N


[doc("Deploy source code to remote")]
deploy-oberon: exec-permissions
    echo "Syncing source-code directory..."
    rsync -azP --delete --exclude=".venv" --exclude="data" --exclude=".mypy_cache" --exclude="notebooks" --exclude=".ruff_cache" --exclude="*.egg-info" "{{current_dir}}/" "{{COML_CLUSTER}}:{{COML_WORKSPACE}}/code/"


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
deploy-jz: exec-permissions
    #!/bin/bash
    # SSHPASS not set
    echo "Syncing source-code directory..."
    if [ -z "${SSHPASS}" ]; then
        rsync -azP --delete --exclude=".venv" --exclude="data" --exclude=".mypy_cache" --exclude="notebooks" --exclude=".ruff_cache" --exclude="src/*.egg-info" "{{current_dir}}/" "{{JZ_CLUSTER}}:{{JZ_SRC_PROD}}"
    else
        sshpass -e rsync -azP --delete --exclude=".venv" --exclude="data" --exclude=".mypy_cache" --exclude="notebooks" --exclude=".ruff_cache" --exclude="src/*.egg-info" "{{current_dir}}/" "{{JZ_CLUSTER}}:{{JZ_SRC_PROD}}"
    fi


[doc("Make executables")]
exec-permissions:
    find src/scripts -name "*.py" -exec chmod +x {} \;
    find experiments -name "*.sh" -exec chmod +x {} \;

[doc("Install module & dependencies")]
install:
    uv sync

[doc("Run Jupyter Server Locally")]
run-notebook:
    uv run jupyter lab

[doc("Check Syntax (RUFF)")]
syntax-check:
    uv run ruff check

syntax-check-file file:
    uv run ruff check {{file}}

[doc("Auto Formatting (RUFF)")]
format:
    ruff format src/lexical_benchmark

check-todo:
    @rg \
    --glob !notebooks/ \
    --glob !justfile \
    --glob !pyproject.toml \
    --ignore-case \
    'fixme|todo|feat' \
    .