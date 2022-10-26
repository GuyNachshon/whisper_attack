#! /bin/bash
DATA=test-clean-100
python run_attack.py attack_configs/rand.yaml --data_csv_name=$DATA --model_label=tiny --root=$RSROOT --sigma=0.09 --save_audio=False --load_audio=False
python run_attack.py attack_configs/rand.yaml --data_csv_name=$DATA --model_label=tiny.en --root=$RSROOT --sigma=0.15 --save_audio=False --load_audio=False
python run_attack.py attack_configs/rand.yaml --data_csv_name=$DATA --model_label=base --root=$RSROOT --sigma=0.13 --save_audio=False --load_audio=False
python run_attack.py attack_configs/rand.yaml --data_csv_name=$DATA --model_label=base.en --root=$RSROOT --sigma=0.16 --save_audio=False --load_audio=False
python run_attack.py attack_configs/rand.yaml --data_csv_name=$DATA --model_label=small --root=$RSROOT --sigma=0.12 --save_audio=False --load_audio=False
python run_attack.py attack_configs/rand.yaml --data_csv_name=$DATA --model_label=small.en --root=$RSROOT --sigma=0.15 --save_audio=False --load_audio=False
python run_attack.py attack_configs/rand.yaml --data_csv_name=$DATA --model_label=medium --root=$RSROOT --sigma=0.13 --save_audio=False --load_audio=False
python run_attack.py attack_configs/rand.yaml --data_csv_name=$DATA --model_label=medium.en --root=$RSROOT --sigma=0.13 --save_audio=False --load_audio=False
python run_attack.py attack_configs/rand.yaml --data_csv_name=$DATA --model_label=large --root=$RSROOT --sigma=0.13 --save_audio=False --load_audio=False
