python main.py \
    --data data/penn \
    --top_model plif_smax \
    --cuda \
    --single_gpu \
    --no_comet \
    --log-interval 100 \
    --lr 20 \
    --bptt 20 \
    --seed 141 \
    --dropouti 0.4 \
    --dropouth 0.3 \
    --nemb 400 \
    --nhid 1150 \
    --switch_epoch 200 \
    --no_analysis \

