
export DECORD_EOF_RETRY_MAX=20480

MODEL_PATH=openai/clip-vit-base-patch32

GPULIST=0,1,2,3,4,5,6,7
N_GPU=8

TEMPORAL_METHOD=longvideo 

CUDA_VISIBLE_DEVICES=$GPULIST
torchrun --nproc_per_node=$N_GPU \
 train.py \
    --resume /path/to/work_dirs/pretrain_coarse/ckpt_ep4.pt \
    --output-dir /path/to/work_dirs/pretrain_fine \
    --model $MODEL_PATH \
    --temporal_method $TEMPORAL_METHOD \
    --lr 1e-7 --wd 0.2 --batch-size 8 --epochs 10 --clip_frames 16 --neg_frames 100