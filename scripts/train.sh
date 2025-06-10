export CUDA_VISIBLE_DEVICES=0,1,2,3

python ../train/optimize.py \
--harmful_dir ../data/train/harmful.json \
--benign_dir ../data/train/benign.json \
--padding 30 \
--proj_weight 1 \
--output_weight 0.02 \
--align_weight 0.0 \
--vector_path ../vector/vector.pt \
--top_threshold 27 \
--bottom_threshold -2 \
--batch_size 2 \
--steps 1200 \
--max_len 1024 \
--model_path ../modellib/LLaVa-1.5-13B \
--save_dir ../train/vsp \
--alpha 1/255
