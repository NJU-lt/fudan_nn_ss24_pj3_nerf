# NeRF
NeRF 3D Reconstruction
# 安装和配置环境
```
conda create -n nerf_mvs python=3.7
conda activate nerf_mvs 
git clone https://github.com/yenchenlin/nerf-pytorch.git
cd nerf-pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```
# 转换LLFF格式数据集
## 下载LLFF源码
```
git clone https://github.com/Fyusion/LLFF.git
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-image
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple imageio
```
## 转换LLFF格式数据集
```
cd LLFF
python imgs2poses.py
```

# NeRF模型训练
```
 python run_nerf.py --config configs/myvideo.txt
```

# NeRF模型测试
```
 python test_psnr.py
```