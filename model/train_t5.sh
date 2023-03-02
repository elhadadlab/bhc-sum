#!/bin/bash
set -e

HF_NAME="google/long-t5-tglobal-base"
EXP="long_t5"
INPUT_LEN=8192
OUTPUT_LEN=384
GRAD_ACCUM=8

python main.py --hf_name $HF_NAME \
  --experiment $EXP --max_input_length $INPUT_LEN \
  --max_output_length $OUTPUT_LEN --grad_accum $GRAD_ACCUM
