#!/bin/bash
#SBATCH --time=0-24:00:00  # d-hh:mm:ss
#SBATCH --account=def-kondrak  # user account
#SBATCH --ntasks=1  # number of task you want to run
#SBATCH --nodes=1  # number of nodes
#SBATCH --cpus-per-task=8  # CPU cores/threads
#SBATCH --gpus-per-node=v100l:1  # GPU cores/threads 
#SBATCH --mem=48000M  # memory per node
#SBATCH --constraint=cascade  # node specification
#SBATCH -J shining  # job name
#SBATCH --mail-type=ALL  # email notification for certain types of events
#SBATCH --mail-user=ning.shi@ualberta.ca  # email address for notification
#SBATCH -e ori_quora_mask_2116_0.error  # error log
#SBATCH -o ori_quora_mask_2116_0.out  # output log

SEED=$(echo '0')  # 0, 1, 2, 3, 4
TASK=$(echo 'ori_quora')  # ori_quora, sep_quora, twitterurl
MASK=$(echo 'True')  # if enable mask control
MODEL=$(echo 'tfm')  # tfm

module load python/3.10.2
source /home/shining/pyvenv/cmput651/bin/activate
cd /home/shining/scratch/cmput651/main
python main.py \
    --task=$TASK \
    --seed=$SEED \
    --mask=$MASK \
    --mask_weights 0.2 0.1 0.1 0.6 \
    --x_x_copy=False \
    --y_x_switch=False \
    --ld=False \
    --lc=False \
    --bt=False \
    --lc_low=2 \
    --lc_compo_size=8 \
    --bt_src_lang=en \
    --bt_tgt_lang=fr \
    --model=$MODEL \
    --encoder=bert-base-uncased \
    --decoder=bert-base-uncased \
    --scorer=deberta-large-mnli \
    --src_translator=opus-mt-ROMANCE-en \
    --tgt_translator=opus-mt-en-ROMANCE \
    --stemming=True \
    --en_max_len=20 \
    --de_max_len=20 \
    --val=True \
    --test=True \
    --train_batch_size=64 \
    --eval_batch_size=512 \
    --num_workers=8 \
    --learning_rate=5e-5 \
    --max_grad_norm=1.0 \
    --weight_decay=0.01 \
    --warmup_steps=5000 \
    --keymetric=ibleu0.8 \
    --val_patience=32 \
    --max_epoch=256 \
    --eval_size=4000 \
    --num_beams=8