sudo apt update
sudo apt install screen
python -m venv ../.venv
source ../.venv/bin/activate
python -m pip install --upgrade pip
pip install jupyter jupyterlab
pip install ipykernel
python -m ipykernel install --user --name=.venv --display-name "Python (.venv)"