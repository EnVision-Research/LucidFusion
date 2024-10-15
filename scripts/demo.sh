#!/usr/bin/env bash
YAML="options/demo.yaml"
DEMO='/hpc2hdd/home/hheat/projects/LGM/data_test/anya_rgba.png'
CKPT='/hpc2hdd/home/hheat/projects/Lucidfusion/output/demo/best.ckpt'
OUTPUT='/hpc2hdd/home/hheat/projects/Lucidfusion/output/res'
# LOG_DIR='stage_1_svd_43k_256_fp32'
export CUDA_VISIBLE_DEVICES='0'
python demo.py --yaml=$YAML --task=demo --data.demo_path=$DEMO --ckpt=$CKPT --save_path=$OUTPUT \
--save_video=true --save_per_view_ply=false --lucid_cam=false \
--test_input_frames=8 --image_data=false --single_input=true --crm=false \
# img_data true -> run imgs in demo, remove_bg for none GSO data
# single_input -> run mv dream
# img_data false, real_data false -> run demo for kun ge's only
# img_data false, real_data true -> run demo for datadir real data!!
# lucid_cam false -> lgm cam
# mv_dream false -> crm diffusion

# /hpc2hdd/JH_DATA/share/yingcongchen/PrivateShareGroup/yingcongchen_datasets/GSO_Val_30/backpack
# /hpc2hdd/home/hheat/projects/gs_shape/temp/gen/kunkun
# /hpc2hdd/home/hheat/projects/LGM/data_test/anya_rgba.png
# /hpc2hdd/home/hheat/projects/CRM/examples/kunkun.webp
# /hpc2hdd/home/hheat/projects/gs_shape/temp/nutella_new