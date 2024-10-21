#!/usr/bin/env bash
YAML="options/demo.yaml"
DEMO='examples/nutella_new'
CKPT='output/demo/best.ckpt'
OUTPUT='output/demo'
MODEL_KEY='pretrained/sd-2-1-base'
export CUDA_VISIBLE_DEVICES='0'
python demo.py --yaml=$YAML --task=demo --data.demo_path=$DEMO --ckpt=$CKPT --save_path=$OUTPUT \
--image_data=true --model_key=$MODEL_KEY \
