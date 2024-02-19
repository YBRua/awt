conda activate torch112
CUDA_VISIBLE_DEVICES=7 python cwm_main_train.py \
 --emsize 768 \
 --msg_len 4 \
 --data ~/code-watermarking/datasets/csn_js \
 --lang javascript \
 --batch_size 32  \
 --epochs 100 \
 --save csn_js_funcs_4bit \
 --optimizer adam \
 --use_reconst_loss \
 --discr_interval 1 \
 --msg_weight 5 \
 --gen_weight 1.5 \
 --reconst_weight 1.5 \
 --scheduler

# CUDA_VISIBLE_DEVICES=6 python cwm_main_train.py \
#  --emsize 768 \
#  --msg_len 2 \
#  --data ~/code-watermarking/data/github_java_funcs \
#  --lang java \
#  --batch_size 32  \
#  --epochs 100 \
#  --save github_java_funcs \
#  --optimizer adam \
#  --use_reconst_loss \
#  --discr_interval 1 \
#  --msg_weight 5 \
#  --gen_weight 1.5 \
#  --reconst_weight 1.5 \
#  --scheduler

# CUDA_VISIBLE_DEVICES=6 python cwm_main_train.py \
#  --emsize 768 \
#  --msg_len 2 \
#  --data ~/code-watermarking/data/gcj_cpp_funcs \
#  --lang cpp \
#  --batch_size 32  \
#  --epochs 100 \
#  --save gcj_cpp_funcs \
#  --optimizer adam \
#  --use_reconst_loss \
#  --discr_interval 1 \
#  --msg_weight 5 \
#  --gen_weight 1.5 \
#  --reconst_weight 1.5 \
#  --scheduler

# CUDA_VISIBLE_DEVICES=6 python cwm_main_train.py \
#  --emsize 768 \
#  --msg_len 2 \
#  --data ~/code-watermarking/data/gcj_java_funcs \
#  --lang java \
#  --batch_size 32  \
#  --epochs 100 \
#  --save gcj_java_funcs \
#  --optimizer adam \
#  --use_reconst_loss \
#  --discr_interval 1 \
#  --msg_weight 5 \
#  --gen_weight 1.5 \
#  --reconst_weight 1.5 \
#  --scheduler
