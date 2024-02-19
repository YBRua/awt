conda activate torch112
CUDA_VISIBLE_DEVICES=4 python csn_main_train.py \
 --msg_len 4 \
 --data data/CodeSearchNet \
 --lang java \
 --batch_size 24  \
 --epochs 60 \
 --save csn_4bit_java_full \
 --resume csn_4bit_java \
 --optimizer adam \
 --use_reconst_loss \
 --use_semantic_loss \
 --sem_weight 6 \
 --use_lm_loss \
 --lm_weight 6 \
 --lm_ckpt csn_java_lm.pt \
 --discr_interval 3 \
 --msg_weight 6 \
 --gen_weight 1 \
 --reconst_weight 2 \
 --scheduler \
 --shared_encoder 1

# CUDA_VISIBLE_DEVICES=5 python csn_main_train.py \
#  --msg_len 4 \
#  --data /home/borui/code-watermarking/datasets/github_c_funcs \
#  --lang cpp \
#  --batch_size 16  \
#  --epochs 60 \
#  --save github_c_baseline_full \
#  --resume github_c_baseline \
#  --optimizer adam \
#  --use_reconst_loss \
#  --use_semantic_loss \
#  --sem_weight 6 \
#  --use_lm_loss \
#  --lm_weight 6 \
#  --lm_ckpt c_func_lm.pt \
#  --discr_interval 3 \
#  --msg_weight 6 \
#  --gen_weight 1 \
#  --reconst_weight 2 \
#  --scheduler \
#  --shared_encoder 1
