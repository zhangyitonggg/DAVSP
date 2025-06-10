python ../vector/extract_hs.py \
--text_dir ../data/vector/harmful.json \
--save_dir ../vector \
--save_name harmful.pt \
--model_path ../modellib/LLaVa-1.5-13B

python ../vector/extract_hs.py \
--text_dir ../data/vector/harmless.json \
--save_dir ../vector \
--save_name harmless.pt \
--model_path ../modellib/LLaVa-1.5-13B


python ../vector/get_layer_avg.py \
--path ../vector/harmful.pt \
--save_path ../vector/harmful_avg.pt \

python ../vector/get_layer_avg.py \
--path ../vector/harmless.pt \
--save_path ../vector/harmless_avg.pt


python ../vector/get_vector.py \
--harmless_path ../vector/harmless_avg.pt \
--harmful_path ../vector/harmful_avg.pt \
--save_path ../vector/vector.pt
