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
    "workspace/code/LexicalBenchmark2"
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


[doc("Make executables")]
exec-permissions:
    find src/scripts -name "*.py" -exec chmod +x {} \;
    find src/ -name "*.sh" -exec chmod +x {} \;

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


_rsync-to source_path target_path:
    #!/usr/bin/env bash
    # SSHPASS not set
    echo "Syncing source-code directory..."
    if [ -z "${SSHPASS}" ]; then
        rsync -azP --delete --exclude=".venv" --exclude="data" --exclude=".mypy_cache" --exclude="notebooks" --exclude=".ruff_cache" --exclude="src/*.egg-info" "{{source_path}}/" "{{target_path}}/"
    else
        sshpass -e rsync -azP --delete --exclude=".venv" --exclude="data" --exclude=".mypy_cache" --exclude="notebooks" --exclude=".ruff_cache" --exclude="src/*.egg-info" "{{source_path}}/" "{{target_path}}/"
    fi


_sync-watcher source_path target_path:
    #!/usr/bin/env bash

    sync_files() {
        echo "Changes noticed @ {{source_path}} | syncing"
        rsync -azh "{{source_path}}" "{{target_path}}" \
            --progress \
            --delete --force \
            --exclude=".venv" \
            --exclude="data" \
            --exclude=".mypy_cache" \
            --exclude="notebooks" \
            --exclude=".ruff_cache" \
            --exclude="*.egg-info"
        return 0
    }
    
    if [ "{{os()}}" = "macos" ]; then
        echo "running using fswatch@macos"
        fswatch -l 10 -e ".git/**" "{{source_path}}" | while read -r changed_path; do sync_files; done
    elif [ "{{os()}}" = "linux" ]; then
        echo "running using inotify-tools@linux"
        while inotifywait -r -e modify,create,delete $source_path
        do
            sync_files 
        done
    fi
    echo "Sync interrupted !!"



[doc("Deploy code to jean-zay : [prod|dev] [single|loop].")]
sync-jz target="dev" mode="sigle":
    #!/bin/bash
    if [ "{{mode}}" = "sigle" ]; then
        if [ "{{target}}" = "prod" ]; then
            echo "RSYNC -> jz@prod"
            just _rsync-to "{{current_dir}}/" "{{JZ_CLUSTER}}:{{JZ_SRC_PROD}}"
        elif [ "{{target}}" = "dev" ]; then
            echo "RSYNC -> jz@dev"
            just _rsync-to "{{current_dir}}/" "{{JZ_CLUSTER}}:{{JZ_SRC_DEV}}"
        else
            echo "Expected target = dev | prod, got {{target}} !!"
        fi
    elif [ "{{mode}}" = "loop" ]; then
        if [ "{{target}}" = "prod" ]; then
            echo "WATCH-RSYNC -> jz@prod"
            just _sync-watcher "{{current_dir}}/" "{{JZ_CLUSTER}}:{{JZ_SRC_PROD}}"
        elif [ "{{target}}" = "dev" ]; then
            echo "WATCH-RSYNC -> jz@dev"
            just _sync-watcher "{{current_dir}}/" "{{JZ_CLUSTER}}:{{JZ_SRC_DEV}}"
        else
            echo "Expected target = dev | prod, got {{target}} !!"
        fi
    else
        echo "Expected mode = single | loop, , got "{{mode}}" !!"
    fi

[doc("Deploy code to oberon (mode=[single|loop]).")]
sync-oberon mode="single":
    #!/bin/bash
    if [ "{{mode}}" = "single" ]; then
        just _rsync-to "{{current_dir}}/" "{{COML_CLUSTER}}:{{COML_WORKSPACE}}/code/"
    elif [ "{{mode}}" = "loop" ]; then
        just _sync-watcher "{{current_dir}}/" "{{COML_CLUSTER}}:{{COML_WORKSPACE}}/code/"
    else
        echo "Expected mode = single | loop, got "{{mode}}" !! "
    fi