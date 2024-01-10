*It takes very long time for benchmark model.*

how to install auto gluon on linux conda environment.

conda create -n ag python=3.10
conda activate ag
conda install -c conda-forge mamba
mamba install -c conda-forge autogluon "pytorch=*=cuda*"

how to run predict.

prepare test_data.csv and train_data.csv in '../../data' folder.
python train.py
modify inference.py to set predictor location.
python inference.py
then you will get submission.csv in outputs folder.