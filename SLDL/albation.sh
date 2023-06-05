python train.py --experiment="albation_1" --gpu=4  --dist_mode="guassian" --use_emf=False --use_fdmrm=False --n_rater=3 --weight 1 0 1 0 &
python train.py --experiment="albation_2" --gpu=4  --dist_mode="guassian" --use_emf=False --use_fdmrm=False --n_rater=3 --weight 1 1 1 0 &
python train.py --experiment="albation_3" --gpu=4  --dist_mode="guassian" --use_emf=False --use_fdmrm=False --n_rater=3 --weight 1 1 1 1 &
python train.py --experiment="albation_4" --gpu=4  --dist_mode="guassian" --use_emf=False --use_fdmrm=True --n_rater=3 --weight 1 1 1 1 &
python train.py --experiment="albation_5" --gpu=5  --dist_mode="guassian" --use_emf=True --use_fdmrm=True --n_rater=3 --weight 1 1 1 1 &
python train.py --experiment="albation_6" --gpu=5  --dist_mode="t_lgd" --use_emf=True --use_fdmrm=True --n_rater=3 --weight 1 1 1 1
