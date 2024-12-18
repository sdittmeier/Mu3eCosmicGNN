conda install pytorch==2.4.0 \
torchvision==0.19.0 \
torchaudio==2.4.0 \
pytorch-cuda=12.1 \
-c pytorch \
-c nvidia && \
pip install --no-cache-dir -r requirements.txt && \
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html \
&& pip install -e . 
#\
#python check_acorn.py