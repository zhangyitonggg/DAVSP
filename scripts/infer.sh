export CUDA_VISIBLE_DEVICES=0,1,2,3

mkdir -p ../result

python ../infer/infer_DAVSP.py \
--text_path ../data/infer/harmful_or_benign.json \
--model_path ../modellib/LLaVa-1.5-13B \
--noise_path ../train/vsp/noise/noise-1200.pt \
--padding 30 \
--result_path ../result/DAVSP_result.json