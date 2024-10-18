#!/usr/bin/env bash
YAML="options/demo.yaml"
DEMO='examples/nutella_new'
CKPT='output/demo/best.ckpt'
OUTPUT='output/demo'
export CUDA_VISIBLE_DEVICES='0'
python demo.py --yaml=$YAML --task=demo --data.demo_path=$DEMO --ckpt=$CKPT --save_path=$OUTPUT \
--save_video=true --save_per_view_ply=false --lucid_cam=false \
--test_input_frames=8 --image_data=true --single_input=false --crm=false \
