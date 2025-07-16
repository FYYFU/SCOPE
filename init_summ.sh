export CUDA_VISIBLE_DEVICES=$1

method=$2 # Support ALLKV, PyramidKV, PyramidInfer SnapKV, H2O, StreamingLLM
max_capacity_prompts=$3
attn_implementation=$4 # Support "flash_attention_2", "sdpa", "eager".
model_path=$5
decoding_metric=$6 # H2O Support None,h2o,(slide, adaptive, discontinuous)---SCOPE
decoding_window_size=$7
decoding_recent_size=$8
save_dir=$9 # path to result save_dir

# source_path=$5
# K=$10 #30,60
# T=$11

python3 run_longbench.py \
    --method ${method} \
    --model_path ${model_path} \
    --max_capacity_prompts ${max_capacity_prompts} \
    --attn_implementation ${attn_implementation} \
    --save_dir ${save_dir} \
    --use_cache True \
    --decoding_window_size ${decoding_window_size} \
    --decoding_recent_size ${decoding_recent_size} \
    --decoding_metric ${decoding_metric} \

    # --max_num_examples ${T} \
    # --K ${K}\
