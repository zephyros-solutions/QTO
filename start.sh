

mkdir kbc/data/FB15k/neural_adj
mkdir kbc/data/FB15k-237/neural_adj
mkdir kbc/data/NELL995/neural_adj
mkdir results

cd kbc
python src/preprocess_datasets.py 

comm_args='--score_rel True --model ComplEx --rank 1000 --learning_rate 0.1 --max_epochs 100'

time CUDA_VISIBLE_DEVICES=0 python src/main.py --dataset FB15k --batch_size 100 --lmbda 0.01 --w_rel 0.1 ${comm_args}
time CUDA_VISIBLE_DEVICES=0 python src/main.py --dataset FB15k-237 --batch_size 1000 --lmbda 0.05 --w_rel 4 ${comm_args}
time CUDA_VISIBLE_DEVICES=0 python src/main.py --dataset NELL995 --batch_size 1000 --lmbda 0.05 --w_rel 0 ${comm_args}

cd ..


time CUDA_VISIBLE_DEVICES=0 python main.py --data_path data/FB15k-betae --kbc_path kbc/data/FB15k/model/best_valid.model --fraction 10 --thrshd 0.001 --neg_scale 6
time CUDA_VISIBLE_DEVICES=0 python main.py --data_path data/FB15k-237-betae --kbc_path kbc/data/FB15k-237/model/best_valid.model --fraction 10 --thrshd 0.0002 --neg_scale 3
time CUDA_VISIBLE_DEVICES=0 python main.py --data_path data/NELL-betae --kbc_path kbc/data/NELL995/model/best_valid.model --fraction 10 --thrshd 0.0002 --neg_scale 6
