python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/augmentation/StratifiedKFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/CO2_Soleimani/augmentation/StratifiedKFold/input_valid_[0-9].csv --feature_names Polymer_Augmented_SMILES,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type RF --hyperparameter_optimization True --hyperparameter_space_path ./co2_hpo_space.json --results_path ../../../training/CO2_Soleimani/augmentation --random_state 22

python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/BRICS/StratifiedKFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/CO2_Soleimani/BRICS/StratifiedKFold/input_valid_[0-9].csv --feature_names Polymer_BRICS,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type RF --hyperparameter_optimization True --hyperparameter_space_path ./co2_hpo_space.json --results_path ../../../training/CO2_Soleimani/BRICS --random_state 22

python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/fingerprint/StratifiedKFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/CO2_Soleimani/fingerprint/StratifiedKFold/input_valid_[0-9].csv --feature_names CO2_FP_radius_3_nbits_512,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type RF --hyperparameter_optimization True --hyperparameter_space_path ./co2_hpo_space.json --results_path ../../../training/CO2_Soleimani/fingerprint --random_state 22

python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_valid_[0-9].csv --feature_names Polymer_manual,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type RF --hyperparameter_optimization True --hyperparameter_space_path ./co2_hpo_space.json --results_path ../../../training/CO2_Soleimani/manual_frag --random_state 22

python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_valid_[0-9].csv --feature_names Polymer_manual_aug,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type RF --hyperparameter_optimization True --hyperparameter_space_path ./co2_hpo_space.json --results_path ../../../training/CO2_Soleimani/manual_frag --random_state 22

python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_valid_[0-9].csv --feature_names Polymer_SMILES,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type RF --hyperparameter_optimization True --hyperparameter_space_path ./co2_hpo_space.json --results_path ../../../training/CO2_Soleimani/manual_frag --random_state 22

python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_valid_[0-9].csv --feature_names Polymer_SELFIES,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type RF --hyperparameter_optimization True --hyperparameter_space_path ./co2_hpo_space.json --results_path ../../../training/CO2_Soleimani/manual_frag --random_state 22

python ../train.py --train_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/CO2_Soleimani/manual_frag/StratifiedKFold/input_valid_[0-9].csv --feature_names Polymer_BigSMILES,T_K,P_Mpa --target_name exp_CO2_sol_g_g --model_type RF --hyperparameter_optimization True --hyperparameter_space_path ./co2_hpo_space.json --results_path ../../../training/CO2_Soleimani/manual_frag --random_state 22