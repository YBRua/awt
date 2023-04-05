conda activate torch112
CUDA_VISIBLE_DEVICES=1 python "csn_sampling_bert.py" \
 --msg_len 4 \
 --vocab_source wikitext \
 --data data/CodeSearchNet \
 --lang java \
 --split test \
 --file_ids 0 \
 --msgs_segment 1 \
 --gen_path ./ckpts/WT2_mt_full_gen_4bit.pt \
 --disc_path ./ckpts/WT2_mt_full_disc_4bit.pt \
 --use_lm_loss \
 --seed 200 \
 --samples_num 10 \
 --dataset_subsample_num 500
