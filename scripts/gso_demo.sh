#!/usr/bin/env bash
YAML="options/demo.yaml"
DEMO='/hpc2hdd/home/hheat/projects/LucidFusion/examples/monkey_chair' #'examples/nutella_new'
CKPT='/hpc2hdd/home/hheat/projects/gs_shape/output/gs_render/gs_sd_98k_wbg_cat_random/best_ep10.ckpt' #'output/demo/best.ckpt'
OUTPUT='output/demo'
export CUDA_VISIBLE_DEVICES='0'
python demo.py --yaml=$YAML --task=demo --data.demo_path=$DEMO --ckpt=$CKPT --save_path=$OUTPUT \
--save_video=true --save_per_view_ply=false --lucid_cam=true \
--test_input_frames=8 --image_data=true --single_input=false --crm=false \

# /hpc2hdd/JH_DATA/share/yingcongchen/PrivateShareGroup/yingcongchen_datasets/GSO_Val_30/backpack
