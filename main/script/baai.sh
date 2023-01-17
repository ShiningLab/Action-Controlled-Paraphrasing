#!/bin/bash
#SBATCH --time=0-16:00:00  # d-hh:mm:ss
#SBATCH --ntasks=1  # number of task you want to run
#SBATCH --nodes=1  # number of nodes
#SBATCH --partition=internal
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1 #request for one 5gb gpu, see manual for more info on this
#SBATCH --job-name=shining  # job name
#SBATCH -e shining_mask_sep_0.error  # error log
#SBATCH -o shining_mask_sep_0.out  # output log

MODEL=$(echo 'tfm')  # tfm
TASK=$(echo 'quora')  # quora
VERSION=$(echo 'mask_sep')  # ori, sep, mask_sep
SEED=$(echo '0')  # 0, 1, 2, 3, 4

cd /home/fujie/shining/repos/cmput651/main
python main.py \
    --seed=$SEED \
    --task=$TASK \
    --version=$VERSION \
    --train_size=100000 \
    --valid_size=4000 \
    --test_size=20000 \
    --encoder=bert-base-uncased \
    --decoder=bert-base-uncased \
    --model=$MODEL \
    --hidden_size=450 \
    --num_hidden_layers=3 \
    --num_attention_heads=9 \
    --intermediate_size=1024 \
    --val=True \
    --test=True \
    --train_batch_size=32 \
    --eval_batch_size=512 \
    --num_workers=12 \
    --learning_rate=5e-5 \
    --max_grad_norm=1.0 \
    --weight_decay=0.01 \
    --warmup_steps=5000 \
    --max_steps=400000 \
    --val_patience=32 \
    --eval_size=4000 \
    --num_beams=8