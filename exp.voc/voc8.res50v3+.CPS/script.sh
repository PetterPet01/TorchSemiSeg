#!/usr/bin/env bash
nvidia-smi

export volna="/root/TorchSemiSeg/"
export NGPUS=1
export OUTPUT_PATH="/root/TorchSemiSeg/output"
export snapshot_dir=$OUTPUT_PATH/snapshot

export batch_size=2
export learning_rate=0.0025
export snapshot_iter=10

export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_PROTO=Simple
python train.py -d 0
# python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py
# export TARGET_DEVICE=$[$NGPUS]
# python eval.py -e 20-34 -d 0-$TARGET_DEVICE --save_path $OUTPUT_PATH/results

# following is the command for debug
# export NGPUS=1
# export batch_size=2
# python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --debug 1