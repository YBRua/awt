conda activate torch112
CUDA_VISIBLE_DEVICES=3 python csn_main_train.py \
 --codebert \
 --emsize 768 \
 --msg_len 4 \
 --data data/CodeSearchNet \
 --lang java \
 --batch_size 80  \
 --epochs 200 \
 --save csn_java_cb_noft \
 --optimizer adam \
 --use_reconst_loss \
 --discr_interval 1 \
 --msg_weight 5 \
 --gen_weight 1.5 \
 --reconst_weight 1.5 \
 --scheduler \
 --allow_cached_dataset \
 --train_subsample_num 36000 \
 --eval_subsample_num 8000 \
