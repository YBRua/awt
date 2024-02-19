conda activate torch112
CUDA_VISIBLE_DEVICES=7 python "cwm_sampling_simplified.py" \
 --msg_len 4 \
 --data ~/code-watermarking/data/github_c_funcs \
 --lang c \
 --msgs_segment 1 \
 --gen_path github_c_funcs_4bit_gen.pt \
 --disc_path github_c_funcs_4bit_disc.pt \
 --seed 200 \
 --samples_num 1 \
 --use_lm_loss
