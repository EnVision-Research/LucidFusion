#!/usr/bin/env bash
YAML="options/demo.yaml"
DEMO='CRM/examples/kunkun.webp'
CKPT='output/demo/best.ckpt'
OUTPUT='output/demo'
export CUDA_VISIBLE_DEVICES='0'
python demo.py --yaml=$YAML --task=demo --data.demo_path=$DEMO --ckpt=$CKPT --save_path=$OUTPUT \
--image_data=false --single_input=true --crm=true --model_key=true \
