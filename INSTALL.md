### Installation

```
1.
conda create --name ita python=3.8 -y
conda activate ita
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c nvidia
pip install -U opencv-python
```

```
2.
git clone https://github.com/facebookresearch/detectron2.git
if error you can:
wget https://github.com/facebookresearch/detectron2/archive/refs/heads/main.zip
and unzip this zip file
cd detectron2-main
pip install -e .
cd ..
```

```
3.
pip install git+https://github.com/cocodataset/panopticapi.git
if error you can:
wget https://github.com/cocodataset/panopticapi/archive/refs/heads/master.zip
and unzip this zip file
cd panopticapi-master/
python setup.py install
cd ..
```

```
4.
git clone https://github.com/bytedance/fcclip.git
if error you can:
wget https://github.com/mcordts/cityscapesScripts/archive/refs/heads/master.zip
and unzip this zip file 
cd cd cityscapesScripts-master/
python setup.py install
cd ..
```

```
pip install -r requirements.txt
```

