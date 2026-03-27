export python3WARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
EVAL_ONLY=False
OVERWRITE=True
QUESTION_TYPE=multi_choice


CKPT=lmms-lab/LLaVA-Video-7B-Qwen2
CONV_MODE=qwen_1_5
VIDEO_DIR=PATH-TO-LongVideoBench/videos
GT_FILE=PATH-TO-LongVideoBench/lvb_val.json
ANNO_PATH=annos/LongVideoBench/siglip_full.json
TEMPORAL_TYPE=oneclip
TOPK=16 


CHUNKS=8
GPULIST=(0 1 2 3 4 5 6 7)


if [ "$OVERWRITE" = False ]; then
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_temporal_${TEMPORAL_TYPE}_${TOPK}_${QUESTION_TYPE}_overwrite_${OVERWRITE}

else
    SAVE_DIR=$(basename $CKPT)_${CONV_MODE}_temporal_${TEMPORAL_TYPE}_${TOPK}_${QUESTION_TYPE}
fi

echo $SAVE_DIR

NUM_GPUS=${#GPULIST[@]}
GPUS_PER_CHUNK=$((NUM_GPUS / CHUNKS))

if [ "$EVAL_ONLY" == False ]; then
    for IDX in $(seq 1 $CHUNKS); do
        START=$(((IDX-1) * GPUS_PER_CHUNK))
        LENGTH=$GPUS_PER_CHUNK
        
        CHUNK_GPUS=(${GPULIST[@]:$START:$LENGTH})
        
        CHUNK_GPUS_STR=$(IFS=,; echo "${CHUNK_GPUS[*]}")
        
        echo "CUDA_VISIBLE_DEVICES=$CHUNK_GPUS_STR"
        CUDA_VISIBLE_DEVICES=$CHUNK_GPUS_STR python3 OneClip-RAG/eval/model_longvideobench.py \
            --model-path $CKPT \
            --video_dir $VIDEO_DIR \
            --gt_file $GT_FILE \
            --output_dir ./work_dirs/eval_longvideobench/$SAVE_DIR \
            --output_name pred \
            --num-chunks $CHUNKS \
            --chunk-idx $(($IDX - 1)) \
            --topk $TOPK \
            --temporal_type $TEMPORAL_TYPE \
            --anno_path $ANNO_PATH \
            --conv-mode $CONV_MODE &
    done

    wait
fi

python3 ./scripts/calculate_score.py \
    --output_dir ./work_dirs/eval_longvideobench/$SAVE_DIR \
    --eval_type $QUESTION_TYPE \
    --num-chunks $CHUNKS
