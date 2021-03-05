CUDA_VISIBLE_DEVICES=2 PYTHONPATH=. python dsgcn/main.py \
    --stage mall \
    --phase test \
    --config dsgcn/configs/Eval_config.yaml \
    --load_from1 data/work_dir/output_models/dsgcn_model_iter_100.pth \
    --save_output \
