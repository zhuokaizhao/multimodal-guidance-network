# script that set up virtual environment dependencies
# Create a virtual env first:
# conda env create --name zoetrope python=3.10
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install torchdata -c pytorch-nightly -y
pip install opencv-python
pip install argparse
pip install tqdm
pip install boto3
pip install ftfy regex
pip install git+https://github.com/openai/CLIP.git
pip install matplotlib
pip install Pillow
pip install -U scikit-learn
pip install pandas
pip install psycopg2-binary
pip install smart-open
pip install transformers