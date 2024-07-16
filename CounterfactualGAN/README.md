# Generating Realistic Counterfactuals for Natural Language Text 
This code accompanies the paper accepted at EMNLP 2021 by Marcel Robeer, Floris Bex and Ad Feelders.

Experiments were performed with Python 3.7.5 on a Tesla V100 GPU with CUDA v10.2. Main package dependencies are `torch==1.6.0`, `torchtext==0.5.0`, `transformers==2.8.0`, `nltk==3.5`, `numpy==1.18.1`, `pandas==1.2.0`, `scikit-learn==0.22.2.post1`, `textstat`, `tensorflow-gpu==1.15`, `tensorflow-hub` and `jupyter==1.0.0`.

#### Datasets
CounterfactualGAN is compared against three baseline methods on three datasets (accessible through `dataset.py`):

Dataset | Class | Task | URL | Folder format
---|---|---|---|---
_Hatespeech_ | `Hatespeech()` | Regresssion | [\[url\]](https://data.world/thomasrdavidson/hate-speech-and-offensive-language) | `hatespeech_data.csv`
_SST-2_ | `SST()` | Binary classification | [\[url\]](https://github.com/clairett/pytorch-sentiment-classification/tree/master/data/SST2) | `SST2/*.tsv`
_SNLI_ | `SNLI()` | Three-class classification | [\[url\]](https://archive.nyu.edu/handle/2451/41728) | `snli_1.0/snli_1.0_*.txt`

The files from the urls above should be downloaded and put into the DATA folder (as defined in `config.py`) according to the folder format in the table, where wildcard `*` are the train/dev/test files. For example, folder `DATA/SST2/` should contain three `.tsv` files, `train.tsv`, `dev.tsv` and `test.tsv`.

#### Baseline methods
Jupyter Notebook `Run Baselines.ipynb` contains the code to determine the fidelity and perceptibility of the baseline methods.

#### Predictive models (black-boxes)
Jupyter Notebook `Run Baselines.ipynb` also contains the code for training and testing the predictive models. By setting the 'Calculate performance' option to True, the trained black-box models (`Whitebox`, `Infersent` and `BERT`) will also produce files in the `/results/` folder that contain the scores (_MSE_ or _F1_) of each method on their test set.

#### Experimental results
Use `Compare methods.ipynb` to compare the quantitative scores obtained for performance, fidelity and perceptibility. In addition, it contains the indices of the test set instances used in the human evaluation.
