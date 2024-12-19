# Utilizing Protein Structure Graph Embeddings to Predict the Pathogenicity of Missense Variants

This repository houses the code for the research paper "Utilizing protein structure graph embeddings to predict the pathogenicity of missense variants", authored by Martin Danner, Matthias Begemann, Miriam Elbracht, Ingo Kurth, and Jeremias Krause. With this code, users can transform pdb protein structures into datasets suitable for machine learning especially to train two autoencoder models: "MPNN Graph Autoencoder" and "GCN Autoencoder". These trained models facilitate the generation of embeddings applicable in downstream analysis, like the prediction of pathogenicity.

You can end-2-end run this code using the sample files and following the steps below. We highly recommend using a machine equipped with a GPU. Please keep in mind that the more layers you add to your graph autoencoders the more compute resources will be needed.

To keep track of your experiments we recommend using [MLFlow](https://mlflow.org/).

If you find this code helpful in your research, please consider citing our paper.

## Compute Resources

In our work within Databricks, we utilized the following setups for predicting Protein Data Bank (PDB) structures of variants and wild types using ESMFold, creating the graph datasets as well as training the models:

1. **MPNN Graph Autoencoder: Atomic-Scoped Dataset**

   - Machine: [Standard NC24ads A100 v4](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nca100v4-series?tabs=sizebasic)
   - Specifications: 220 GB Memory, 1 GPU
   - Databricks Runtime Version: 15.4 LTS ML (includes Apache Spark 3.5.0, GPU support, Scala 2.12)

2. **GCN Autoencoder and XGBoost Classifier: Residue-Scoped Dataset**
   - Machine: [Standard NV36ads A10 v5](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/gpu-accelerated/nvadsa10v5-series?tabs=sizebasic)
   - Specifications: 440 GB Memory, 1 GPU
   - Databricks Runtime Version: 15.4 LTS ML (includes Apache Spark 3.5.0, GPU support, Scala 2.12)

## Getting Started

### Prerequisites

- Python 3.12
- Optional: Make for leveraging the Makefile

### Installation and Usage

To quickly start using this project, follow these steps. You can test them with the provided [sample structures](./src/data/sample_structures/), [sample folds](./src/data/sample_folds/) and default [configurations](./src/configs/).

1. (Optional): Installation of Make

   macOS using Homebrew:

   ```
   brew install make
   ```

   Linux:

   ```
   sudo apt update
   sudo apt install make
   ```

   Windows using Chocolatey:

   ```
   choco install make
   ```

2. Clone the repository in your local environment.

3. Navigate to the project root directory using the terminal.

4. Execute the following command, which uses a [Makefile](./Makefile)
   to create a Python virtual environment (`venv`), activates the `venv`, and then installs necessary dependencies from the [requirements.txt](./requirements.txt) file. NOTE: You can also just copy the commands from the Makefile and run it in your terminal.

   ```
   make create_env
   ```

5. Activate your venv

   macOS and Linux:

   ```
   source venv/bin/activate
   ```

   Windows:

   ```
   .\venv\Scripts\activate
   ```

6. You can manipulate the training and inference configurations as per your needs using the provided yaml files ([gcn_config.yaml](./src/configs/gcn_config.yaml), [mpnn_config.yaml](./src/configs/mpnn_config.yaml)).

7. To train the Residue Autoencoder, run:

   ```
   make train_residue_autoencoder
   ```

8. Run inference on the trained Residue Autoencoder by executing:

   ```
   make run_inference_residue_autoencoder
   ```

9. Similarly, to train and run inference on the Atomic Autoencoder, run the following commands respectively:
   
   ```
   make train_atomic_autoencoder
   ```
   
   ```
   make run_inference_atomic_autoencoder
   ```

10. To make use of the embeddings for pathogenicity prediction using an Optuna optimized XGBoost Classifier feel free look into [optimize_train_and_run_xgboost.py](./src/optimize_train_and_run_xgboost.py) & [xgboost_config.yaml](./src/configs/xgboost_config.yaml) or run script directly with sample folds via:
   
      ```
      make optimize_train_and_run_xgboost
      ```

### Citation

If you use the tool presented in this repository, please cite us:

```
Utilizing protein structure graph embeddings to predict the pathogenicity of missense variants 
Martin Danner, Matthias Begemann, Miriam Elbracht, Ingo Kurth, Jeremias Krause
```
