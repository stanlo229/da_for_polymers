model_types=('RF' 'BRT' 'SVM')
for model in "${model_types}"
do
    # AUGMENTED SMILES
    python ../train.py --train_path ../../../data/input_representation/OPV_Min/augmentation/KFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/OPV_Min/augmentation/KFold/input_valid_[0-9].csv --feature_names Augmented_SMILES --target_name calc_PCE_percent --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./opv_hpo_space.json --results_path ../../../training/OPV_Min/Augmented_SMILES --random_state 22
    
    # BRICS
    # python ../train.py --train_path ../../../data/input_representation/OPV_Min/BRICS/KFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/OPV_Min/BRICS/KFold/input_valid_[0-9].csv --feature_names DA_pair_BRICS --target_name calc_PCE_percent --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./opv_hpo_space.json --results_path ../../../training/OPV_Min/BRICS --random_state 22
    
    # FINGERPRINT
    # python ../train.py --train_path ../../../data/input_representation/OPV_Min/fingerprint/KFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/OPV_Min/fingerprint/KFold/input_valid_[0-9].csv --feature_names DA_FP_radius_3_nbits_512 --target_name calc_PCE_percent --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./opv_hpo_space.json --results_path ../../../training/OPV_Min/fingerprint --random_state 22
    
    # MANUAL FRAG
    # python ../train.py --train_path ../../../data/input_representation/OPV_Min/manual_frag/KFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/OPV_Min/manual_frag/KFold/input_valid_[0-9].csv --feature_names DA_manual --target_name calc_PCE_percent --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./opv_hpo_space.json --results_path ../../../training/OPV_Min/manual_frag --random_state 22
    
    # AUGMENTED MANUAL FRAG
    python ../train.py --train_path ../../../data/input_representation/OPV_Min/manual_frag/KFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/OPV_Min/manual_frag/KFold/input_valid_[0-9].csv --feature_names DA_manual_aug --target_name calc_PCE_percent --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./opv_hpo_space.json --results_path ../../../training/OPV_Min/manual_frag_aug --random_state 22
    
    # SMILES
    # python ../train.py --train_path ../../../data/input_representation/OPV_Min/SMILES/KFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/OPV_Min/SMILES/KFold/input_valid_[0-9].csv --feature_names DA_SMILES --target_name calc_PCE_percent --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./opv_hpo_space.json --results_path ../../../training/OPV_Min/SMILES --random_state 22
    
    # SELFIES
    # python ../train.py --train_path ../../../data/input_representation/OPV_Min/SMILES/KFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/OPV_Min/SMILES/KFold/input_valid_[0-9].csv --feature_names DA_SELFIES --target_name calc_PCE_percent --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./opv_hpo_space.json --results_path ../../../training/OPV_Min/SELFIES --random_state 22
    
    # BIGSMILES
    # python ../train.py --train_path ../../../data/input_representation/OPV_Min/SMILES/KFold/input_train_[0-9].csv --validation_path ../../../data/input_representation/OPV_Min/SMILES/KFold/input_valid_[0-9].csv --feature_names DA_BigSMILES --target_name calc_PCE_percent --model_type "$model" --hyperparameter_optimization True --hyperparameter_space_path ./opv_hpo_space.json --results_path ../../../training/OPV_Min/BigSMILES --random_state 22
done