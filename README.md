# NUS EE5442 Project - Milestone 1

Requirements: pandas, pytorch

To run an experiment:
- Download datasets: `python dataset_downloaders.py`
- Train models: select dataset and then `python train_cuda.py` (or `python train_mps.py` if on Apple Silicon)
- Run: `python experiments_quantization.py`

Results of experiments are stored in CSV format in the `results` folder.

WARNING: never push datasets in the remote repo!