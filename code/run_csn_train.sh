conda activate torch112
echo "yes, it's csn_main_train.py"
CUDA_VISIBLE_DEVICES=3 python csn_main_train.py \
 --emsize 768 \
 --msg_len 4 \
 --data data/CodeSearchNet \
 --lang java \
 --batch_size 32  \
 --epochs 60 \
 --save csn_4bit_java \
 --optimizer adam \
 --use_reconst_loss \
 --discr_interval 1 \
 --msg_weight 5 \
 --gen_weight 1.5 \
 --reconst_weight 1.5 \
 --scheduler

#  --data data/CodeSearchNet \

# conda activate torch112
# echo "yes, it's csn_main_train.py"
# CUDA_VISIBLE_DEVICES=0 python csn_main_train.py \
#  --emsize 768 \
#  --msg_len 4 \
#  --data data/CodeSearchNet \
#  --lang java \
#  --batch_size 32  \
#  --epochs 60 \
#  --save csn_java_recon3 \
#  --optimizer adam \
#  --use_reconst_loss \
#  --discr_interval 1 \
#  --msg_weight 5 \
#  --gen_weight 1.5 \
#  --reconst_weight 1.5 \
#  --scheduler
