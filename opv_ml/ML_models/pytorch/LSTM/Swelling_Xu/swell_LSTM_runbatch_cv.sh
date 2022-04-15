#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --output=/project/6025683/stanlo/opv_ml/opv_ml/ML_models/pytorch/LSTM/Swelling_Xu/slurm_batch.out
#SBATCH --error=/project/6025683/stanlo/opv_ml/opv_ml/ML_models/pytorch/LSTM/Swelling_Xu/slurm_batch.err
#SBATCH --account=def-aspuru
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=24G
module load python
source /project/6025683/stanlo/opv_project/bin/activate
python swell_LSTM_batch_cv.py --n_hidden 256 --n_embedding 256 --drop_prob 0.3 --learning_rate 1e-2 --train_batch_size 256