#!/bin/bash
set -e
CUR_DIR=$(cd $(dirname $0);pwd)
BERT_DIR=$(cd ${CUR_DIR}/../;pwd)

pushd $BERT_DIR
if [ -z $BERT_INFER_MODEL ]; then
    echo "[ERROR] Please set BERT_INFER_MODEL."
    exit 1
fi

if [ -z $SQUAD_DIR ]; then
    echo "[ERROR] Please set SQUAD_DIR."
    exit 1
fi
bash models/scripts/run_squad.sh \
$BERT_INFER_MODEL \
2 \
4 \
0.00003 \
fp32 \
4 \
1 \
$SQUAD_DIR \
checkpoints/cased_L-12_H-768_A-12/vocab.txt \
output \
eval \
checkpoints/cased_L-12_H-768_A-12/bert_config.json \
-1 \
-1 \
-1 \
O0 \
mlu

popd
