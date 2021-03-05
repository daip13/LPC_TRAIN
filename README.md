# LPC_train
This is the code for LPC model training

## Setup
1. Clone the enter this repository:
```
https://github.com/daip13/LPC_train.git
```

2. Create a docker image for this project:
    We use the same environments as in [Learning to Cluster Faces](https://github.com/yl-1993/learn-to-cluster)

3. Generate the training data for model training.
```
cd /root/LPC_train/traindata_generation/
python traindata_generation.py --detection_file_path /root/LPC_train/dataset/MOT17/results_reid_with_traindata/detection/ --gt_file_path /root/LPC_train/dataset/MOT17/train/ --num_proc 5 --output_path /root/LPC_train/dataset/MOT17/results_reid_with_traindata/lpc_traindata/
```

4. Do model training
```
1. Split the dataset to train data and test data by using the leave-one-out strategy.
2. Change the train_data and test_data path in /root/LPC_train/dsgcn/configs/config.yaml
3. cd /root/LPC_train/ && sh main.sh
```

Notice: the training data for MOT17 is very small, so the model is easy to be overfitting. Please choose the best trained model by evaluating its performance on the test set. For MOT17, we use the model trained with 100 iterations. 
