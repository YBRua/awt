conda activate torch112
CUDA_VISIBLE_DEVICES=1 python lm_train.py \
  --epochs 750 \
  --data data/CodeSearchNet \
  --lang java \
  --save csn_lm.pt \
  --dropouth 0.2 \
  --seed 1882 \
  --bptt 120 \
  --train_source train_36000.json \
  --valid_source valid_8000.json \
  --test_source test_8000.json  \
