bash init_summ.sh \
    2 \
    pyramidkv \
    64 \
    flash_attention_2 \
    meta-llama/Llama-3.1-8B-Instruct \
    None \
    1 \
    0 \
    ./results/


bash init_summ.sh \
    2 \
    pyramidkv \
    128 \
    flash_attention_2 \
    meta-llama/Llama-3.1-8B-Instruct \
    None \
    1 \
    0 \
    ./results/

bash init_summ.sh \
    2 \
    pyramidkv \
    256 \
    flash_attention_2 \
    meta-llama/Llama-3.1-8B-Instruct \
    None \
    1 \
    0 \
    ./results/

bash init_summ.sh \
    2 \
    pyramidkv \
    512 \
    flash_attention_2 \
    meta-llama/Llama-3.1-8B-Instruct \
    None \
    1 \
    0 \
    ./results/

bash init_summ.sh \
    2 \
    pyramidkv \
    1024 \
    flash_attention_2 \
    meta-llama/Llama-3.1-8B-Instruct \
    None \
    1 \
    0 \
    ./results/
