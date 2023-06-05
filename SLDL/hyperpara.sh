python train.py --experiment="nrater_1" --gpu=1  --n_rater=1 &
python train.py --experiment="nrater_2" --gpu=1  --n_rater=2 &
python train.py --experiment="nrater_3" --gpu=1  --n_rater=3 &
python train.py --experiment="nrater_4" --gpu=2  --n_rater=4 &
python train.py --experiment="nrater_5" --gpu=2  --n_rater=5 &
python train.py --experiment="nrater_6" --gpu=3  --n_rater=6 &
# python train.py --experiment="kl_weight_0" --gpu=1  --weight 1 0 1 1 &
# python train.py --experiment="kl_weight_0.01" --gpu=1  --weight 1 0.01 1 1 &
# python train.py --experiment="kl_weight_0.1" --gpu=1  --weight 1 0.1 1 1 &
# python train.py --experiment="kl_weight_1" --gpu=2  --weight 1 1 1 1 &
# python train.py --experiment="kl_weight_10" --gpu=2  --weight 1 10 1 1 &
# python train.py --experiment="kl_weight_100" --gpu=3  --weight 1 100 1 1
# python train.py --experiment="uni_weight_0" --gpu=5  --weight 1 1 1 0 &
# python train.py --experiment="uni_weight_0.01" --gpu=5  --weight 1 1 1 0.01 &
# python train.py --experiment="uni_weight_0.1" --gpu=5  --weight 1 1 1 0.1 &
# python train.py --experiment="uni_weight_1" --gpu=1  --weight 1 1 1 1 &
# python train.py --experiment="uni_weight_10" --gpu=1  --weight 1 1 1 10 &
# python train.py --experiment="uni_weight_100" --gpu=1  --weight 1 1 1 100 &
# python train.py --experiment="sigma_0.01" --gpu=6  --sigma 0.01 &
# python train.py --experiment="sigma_0.02" --gpu=6  --sigma 0.02 &
# python train.py --experiment="sigma_0.04" --gpu=6  --sigma 0.04 &
# python train.py --experiment="sigma_0.08" --gpu=7  --sigma 0.08 &
# python train.py --experiment="sigma_0.16" --gpu=5  --sigma 0.16 &
# python train.py --experiment="sigma_0.32" --gpu=6  --sigma 0.32 