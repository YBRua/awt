conda activate torch112
# CUDA_VISIBLE_DEVICES=7 python "csn_sampling_bert.py" \
#  --msg_len 4 \
#  --vocab_source ./data/CodeSearchNet/javascript/train_0.json\
#  --data /home/borui/code-watermarking/datasets/csn_js \
#  --lang javascript \
#  --split test \
#  --msgs_segment 1 \
#  --gen_path ./csn_js_4bit_full_gen.pt \
#  --disc_path ./csn_js_4bit_full_disc.pt \
#  --use_lm_loss \
#  --lm_ckpt ./csn_js_lm.pt \
#  --seed 42 \
#  --samples_num 1

CUDA_VISIBLE_DEVICES=3 python "csn_sampling_bert.py" \
 --msg_len 4 \
 --vocab_source /mnt/disk3/borui/awt-orig/code/data/CodeSearchNet/javascript/train_0.json\
 --data ./data/CodeSearchNet/javascript \
 --lang javascript \
 --split test \
 --msgs_segment 1 \
 --gen_path ./csn_js_4bit_full_gen.pt \
 --disc_path ./csn_js_4bit_full_disc.pt \
 --use_lm_loss \
 --lm_ckpt ./csn_js_lm.pt \
 --seed 42 \
 --samples_num 1

# | java-test | msg_acc: 0.7323 | bit_acc: 0.9226 | encode_time: 0.2076 | decode_time: 0.0041 | pipeline_time: 0.2117 | forward_sent_enc_time: 0.0035 | tot_sampling_time: 0.1514 | tot_gan_time: 0.0096 | tot_beam_search_time: 0.0430 |
# | javascript-test | msg_acc: 0.6657 | bit_acc: 0.8925 | encode_time: 0.2275 | decode_time: 0.0045 | pipeline_time: 0.2320 | forward_sent_enc_time: 0.0038 | tot_sampling_time: 0.1694 | tot_gan_time: 0.0087 | tot_beam_search_time: 0.0456 |

# CUDA_VISIBLE_DEVICES=5 python "csn_sampling_bert.py" \
#  --msg_len 4 \
#  --vocab_source wikitext \
#  --data /home/borui/code-watermarking/datasets/csn_js \
#  --lang javascript \
#  --split test \
#  --msgs_segment 1 \
#  --gen_path ./ckpts/WT2_mt_full_gen_4bit.pt \
#  --disc_path ./ckpts/WT2_mt_full_disc_4bit.pt \
#  --use_lm_loss \
#  --lm_ckpt ./ckpts/WT2_lm.pt \
#  --seed 42 \
#  --samples_num 1

# CUDA_VISIBLE_DEVICES=7 python "csn_sampling_bert.py" \
#  --msg_len 4 \
#  --vocab_source /home/borui/code-watermarking/datasets/github_java_funcs/train_0.json\
#  --data /home/borui/code-watermarking/datasets/github_java_funcs \
#  --lang java \
#  --split test \
#  --msgs_segment 1 \
#  --gen_path ./github_java_baseline_full_gen.pt \
#  --disc_path ./github_java_baseline_full_disc.pt \
#  --use_lm_loss \
#  --lm_ckpt ./csn_lm.pt \
#  --seed 42 \
#  --samples_num 1
