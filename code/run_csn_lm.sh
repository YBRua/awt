conda activate torch112
CUDA_VISIBLE_DEVICES=3 python lm_train.py \
  --epochs 100 \
  --data data/CodeSearchNet/java \
  --lang java \
  --save csn_java_lm.pt \
  --dropouth 0.2 \
  --seed 1882 \
  --bptt 120 \
  --train_source train_0.json \
  --valid_source valid_0.json \
  --test_source test_0.json
