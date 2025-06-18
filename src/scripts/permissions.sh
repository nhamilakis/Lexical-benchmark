#!/bin/bash

if [ "$(hostname -d)" = "oberon2" ]; then
    echo "is on oberon2"
    echo "applying permissions..."
    chown -R :bootphon $PROJECT/lexical-benchmark/v2
    chmod -R g+rw $PROJECT/lexical-benchmark/v2
    echo "permissions set on $PROJECT/lexical-benchmark/v2"
elif [[ "$(hostname)" == *"jean-zay"* ]]; then
    echo "is on jean-zay"
    echo "applying permissions..."
    chown -R :hhb $ALL_CCFRWORK/lexical-benchmark
    chmod -R g+rw $ALL_CCFRWORK/lexical-benchmark
    echo "permissions set on $ALL_CCFRWORK/lexical-benchmark"
else
    echo "Server not detected"
fi

