conda activate torch112
CUDA_VISIBLE_DEVICES=2 python "csn_sampling_lm.py" \
 --msg_len 4 \
 --codebert \
 --vocab_source ./data/CodeSearchNet/java/train_36000.json \
 --data data/CodeSearchNet \
 --lang java \
 --split test \
 --file_ids 0 \
 --msgs_segment 1 \
 --gen_path csn_java_cb_noft_gen.pt \
 --disc_path csn_java_cb_noft_disc.pt \
 --use_lm_loss \
 --seed 200 \
 --samples_num 10 \
 --dataset_subsample_num 500
