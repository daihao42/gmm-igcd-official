#!/bin/bash 

#SBATCH --account=ecsstaff
#SBATCH --partition=ecsall
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=72:00:00

module load gcc

python main.py --base 50 --increment 10 --pretrained_model_name dino-vitb16 --data_dir datasets/cifar100 --trail_name happy_mix_increment_mngmm_dinovb16_cifar_100
python main.py --base 50 --increment 10 --pretrained_model_name dino-vitb16-sl-s0 --data_dir datasets/cifar100 --trail_name happy_mix_increment_mngmm_dinovb16_sl_cifar_100
python main.py --base 100 --increment 20 --pretrained_model_name dino-vitb16 --data_dir datasets/tinyimagenet --num_classes 200 --trail_name happy_mix_increment_mngmm_dinovb16_tiny_imagenet
python main.py --base 50 --increment 10 --pretrained_model_name dino-vitb16 --data_dir datasets/imagenet100 --trail_name happy_mix_increment_mngmm_dinovb16_imagenet100
