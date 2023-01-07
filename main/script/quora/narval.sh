#!/bin/bash
#SBATCH --time=0-16:00:00  # d-hh:mm:ss
#SBATCH --account=def-kondrakb  # user account
#SBATCH --ntasks=1  # number of task you want to run
#SBATCH --nodes=1  # number of nodes
#SBATCH --cpus-per-task=12  # CPU cores/threads
#SBATCH --gpus-per-node=a100:1  # Number of GPU(s) per node
#SBATCH --mem=127500M  # memory per node
#SBATCH -J shining  # job name
#SBATCH --mail-type=ALL  # email notification for certain types of events
#SBATCH --mail-user=ning.shi@ualberta.ca  # email address for notification
#SBATCH -e twitterurl_mask_0.error  # error log
#SBATCH -o twitterurl_mask_0.out  # output log

MODEL=$(echo 'tfm')  # tfm
TASK=$(echo 'twitterurl')  # ori_quora, sep_quora, twitterurl
SEED=$(echo '0')  # 0, 1, 2, 3, 4
MASK=$(echo 'True') # if enable mask control

module load python/3.10.2
source /home/shining/pyvenv/cmput651/bin/activate
cd /home/shining/scratch/cmput651/main
python main.py \
    --task=$TASK \
    --seed=$SEED \
    --mask=$MASK \
    --mask_weights 0.8 0.1 0.1 \
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
    --hidden_size=450 \
    --num_hidden_layers=3 \
    --num_attention_heads=9 \
    --intermediate_size=1024 \
    --stemming=True \
    --en_max_len=20 \
    --de_max_len=20 \
    --val=True \
    --test=True \
    --train_batch_size=64 \
    --eval_batch_size=512 \
    --num_workers=12 \
    --learning_rate=5e-5 \
    --max_grad_norm=1.0 \
    --weight_decay=0.01 \
    --warmup_steps=5000 \
    --keymetric=ibleu0.8 \
    --val_patience=32 \
    --eval_size=4000 \
    --num_beams=8