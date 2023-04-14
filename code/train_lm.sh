conda activate torch112
CUDA_VISIBLE_DEVICES=6 python lm_train.py \
  --codebert \
  --epochs 750 \
  --data data/CodeSearchNet \
  --lang java \
  --save csn_lm.pt \
  --dropouth 0.2 \
  --seed 1882 \
  --bptt 120 \
  --train_source train_36000_codebert.json \
  --valid_source valid_8000_codebert.json \
  --test_source test_8000_codebert.json  \
