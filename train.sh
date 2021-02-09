CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
PYTHONPATH=. python dsgcn/main.py \
    --stage mall \
    --phase train \
    --config dsgcn/configs/config.yaml \
    --work_dir ./data/work_dir/output_models/ 
