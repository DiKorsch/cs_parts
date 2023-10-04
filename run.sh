#!/usr/bin/env bash
export DATA=${DATA:-data_info.yml}
DATASET=${DATASET:-CUB200}
MODEL_TYPE=${MODEL_TYPE:-cvmodelz.InceptionV3}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}

_now=$(date +%Y-%m-%d-%H.%M.%S.%N)
OUTPUT_PREFIX=${OUTPUT_PREFIX:-"outputs"}
OUTPUT_DIR=${OUTPUT_DIR:-${OUTPUT_PREFIX}/${DATASET}/${MODEL_TYPE}/${_now}}

OPTS="${OPTS} --model_type ${MODEL_TYPE}"
OPTS="${OPTS} --weights $(realpath ${WEIGHTS:-models/ft_${DATASET}_inception.npz})"
OPTS="${OPTS} --pretrained_on ${PRETRAIN:-inat}"
OPTS="${OPTS} --prepare_type ${PREPARE_TYPE:-model}"
OPTS="${OPTS} --input_size ${INPUT_SIZE:-299}"
OPTS="${OPTS} --label_shift ${LABEL_SHIFT:-1}"
OPTS="${OPTS} --n_jobs ${N_JOBS:-6}"
OPTS="${OPTS} --gpu ${GPU:-0}"
OPTS="${OPTS} --center_crop_on_val"
# OPTS="${OPTS} --no_dump"
OPTS="${OPTS} --output ${OUTPUT_DIR}"

VACUUM=${VACUUM:-1}
if [[ $VACUUM == 1 ]]; then
    echo "=!=!=!= On error, removing folder ${OUTPUT_DIR} =!=!=!="
    OPTS="${OPTS} --vacuum"
fi

${PYTHON:-python} main.py \
    ${DATA} \
    ${DATASET} \
    GLOBAL \
    $OPTS $@

# { # try
# } || { # catch

#     if [[ ${VACUUM} == 1 ]]; then
#         echo "Error occured! Removing ${OUTPUT_DIR}"
#         rm -r ${OUTPUT_DIR}
#     fi
# }
