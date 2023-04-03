CUDA_VISIBLE_DEVICES=1 python "evaluate_sampling_$1.py" \
 --msg_len 4 \
 --data data/wikitext-2 \
 --bptt 80 \
 --msgs_segment 5 \
 --gen_path ./WT2_mt_full_gen_4bit.pt \
 --disc_path ./WT2_mt_full_disc_4bit.pt \
 --use_lm_loss 1 \
 --seed 200 \
 --samples_num 10
