conda activate torch112
CUDA_VISIBLE_DEVICES=5 python lm_train_cwm.py \
  --epochs 100 \
  --data ~/code-watermarking/data/github_c \
  --lang c \
  --save github_c_lm.pt \
  --dropouth 0.2 \
  --seed 1882 \
  --bptt 256 \
