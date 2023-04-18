conda activate torch112
CUDA_VISIBLE_DEVICES=1 python "cwm_sampling_simplified.py" \
 --msg_len 4 \
 --data ~/code-watermarking/data/github_c \
 --lang c \
 --msgs_segment 1 \
 --gen_path github_c_gen.pt \
 --disc_path github_c_disc.pt \
 --seed 200 \
 --samples_num 1 \
