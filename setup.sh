module load cuda/10.2 python/3.7
cp -r data/ /tmp/
virtualenv --no-download /tmp/x
source /tmp/x/bin/activate
pip install --no-index -r requirements.txt