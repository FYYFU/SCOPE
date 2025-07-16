bash init_summ.sh \
    4 \
    SnapKV \
    64 \
    flash_attention_2 \
    meta-llama/Llama-3.1-8B-Instruct \
    slide \
    256 \
    128 \
    ./results/


bash init_summ.sh \
    4 \
    SnapKV \
    128 \
    flash_attention_2 \
    meta-llama/Llama-3.1-8B-Instruct \
    slide \
    256 \
    128 \
    ./results/

bash init_summ.sh \
    4 \
    SnapKV \
    256 \
    flash_attention_2 \
    meta-llama/Llama-3.1-8B-Instruct \
    slide \
    256 \
    128 \
    ./results/

bash init_summ.sh \
    4 \
    SnapKV \
    512 \
    flash_attention_2 \
    meta-llama/Llama-3.1-8B-Instruct \
    slide \
    256 \
    128 \
    ./results/

bash init_summ.sh \
    4 \
    SnapKV \
    1024 \
    flash_attention_2 \
    meta-llama/Llama-3.1-8B-Instruct \
    slide \
    256 \
    128 \
    ./results/