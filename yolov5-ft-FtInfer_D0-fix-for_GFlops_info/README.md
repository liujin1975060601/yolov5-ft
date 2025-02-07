# rotate-yolov5


## Install DOTA_Devkit
### 如果在window要根据情况,转换一下cpp的编码格式再进行编译
```
sudo apt-get install swig
cd DOTA_devkit/polyiou
swig -c++ -python csrc/polyiou.i
python setup.py build_ext --inplace

```