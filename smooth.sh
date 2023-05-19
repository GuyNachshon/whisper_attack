#!/bin/bash
LOAD=False
DATA=test-clean-100
NAME=pgd
NBITER=200
SEED=1000

for SNR in 10 15 20 25 30 35 40 45 50
do
    python run_attack.py attack_configs/whisper/pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base.en --nb_iter=100 --load_audio=$LOAD --seed=$SEED --attack_name=$NAME  --snr=$SNR
    python run_attack.py attack_configs/whisper/pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=base.en.smooth.02 --nb_iter=200 --load_audio=$LOAD --seed=$SEED --attack_name=$NAME  --snr=$SNR
done
    
for SNR in 10 15 20 25 30 35 40 45 50
do
    python run_attack.py attack_configs/whisper/pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=small.en --nb_iter=100 --load_audio=$LOAD --seed=$SEED --attack_name=$NAME  --snr=$SNR
    python run_attack.py attack_configs/whisper/pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=small.en.smooth.02 --nb_iter=200 --load_audio=$LOAD --seed=$SEED --attack_name=$NAME  --snr=$SNR
done


for SNR in 10 15 20 25 30 35 40 45 50
do
    python run_attack.py attack_configs/whisper/pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=medium.en --nb_iter=100 --load_audio=$LOAD --seed=$SEED --attack_name=$NAME  --snr=$SNR
    python run_attack.py attack_configs/whisper/pgd.yaml --root=$RSROOT --data_csv_name=$DATA --model_label=medium.en.smooth.02 --nb_iter=200 --load_audio=$LOAD --seed=$SEED --attack_name=$NAME  --snr=$SNR
done
    
    

