conda activate torch112
CUDA_VISIBLE_DEVICES=3 python "csn_sampling_lm.py" \
 --msg_len 4 \
 --codebert \
 --vocab_source ./data/CodeSearchNet/java/train_36000.json \
 --data /home/liwei/csy/Code-Text/code-to-text/dataset \
 --lang java \
 --split test \
 --msgs_segment 1 \
 --gen_path csn_java_fullgen.pt \
 --disc_path csn_java_full_disc.pt \
 --use_lm_loss \
 --lm_ckpt csn_lm.pt \
 --seed 200 \
 --samples_num 10 \
