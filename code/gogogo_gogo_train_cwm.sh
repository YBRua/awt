conda activate torch112
CUDA_VISIBLE_DEVICES=5 python cwm_main_train.py \
 --emsize 768 \
 --msg_len 4 \
 --data ~/code-watermarking/data/github_c \
 --lang c \
 --batch_size 32  \
 --epochs 100 \
 --save github_c \
 --optimizer adam \
 --use_reconst_loss \
 --discr_interval 1 \
 --msg_weight 5 \
 --gen_weight 1.5 \
 --reconst_weight 1.5 \
 --scheduler
