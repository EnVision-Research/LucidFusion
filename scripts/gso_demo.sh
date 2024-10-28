#!/usr/bin/env bash
YAML="options/demo.yaml"
DEMO='examples/chicken'
OUTPUT='output/demo'
MODEL_KEY='pretrained/'
export CUDA_VISIBLE_DEVICES='0'
python demo.py --yaml=$YAML --task=demo --data.demo_path=$DEMO --save_path=$OUTPUT \
--model_key=$MODEL_KEY \
