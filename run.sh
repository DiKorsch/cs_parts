#!/usr/bin/env bash
DATASET=${DATASET:-CUB200}

OPTS="${OPTS} --model_type ${MODEL_TYPE:-cvmodelz.InceptionV3}"
OPTS="${OPTS} --weights $(realpath ${WEIGHTS:-models/ft_${DATASET}_inception.npz})"
OPTS="${OPTS} --pretrained_on ${PRETRAIN:-inat}"
OPTS="${OPTS} --prepare_type ${PREPARE_TYPE:-model}"
OPTS="${OPTS} --input_size ${INPUT_SIZE:-427}"
OPTS="${OPTS} --label_shift ${LABEL_SHIFT:-1}"
OPTS="${OPTS} --gpu ${GPU:-0}"
OPTS="${OPTS} --center_crop_on_val"

${PYTHON:-python} main.py \
      ${DATA:-data_info.yml} \
      ${DATASET} \
      GLOBAL \
      $OPTS $@
