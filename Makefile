create_env:
	python3.12 -m venv venv; . venv/bin/activate; pip install -r requirements.txt

train_residue_autoencoder:
	 python './src/training_gae.py' --config_path './src/configs/gcn_config.yaml'

run_inference_residue_autoencoder:
	 python './src/inference_gae.py' --config_path './src/configs/gcn_config.yaml'

train_atomic_autoencoder:
	 python './src/training_gae.py' --config_path './src/configs/mpnn_config.yaml'

run_inference_atomic_autoencoder:
	 python './src/inference_gae.py' --config_path './src/configs/mpnn_config.yaml'

optimize_train_and_run_xgboost:
	 python './src/optimize_train_and_run_xgboost.py' --config_path './src/configs/xgboost_config.yaml'