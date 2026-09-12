#!/bin/bash  

# 切换到脚本所在的目录  
cd "$(dirname "$0")"
# 删除当前目录下的所有.pyd和.cpp文件  
rm -f *.so
rm -f poly_cpu.cpp
# 递归删除build目录及其所有内容  
rm -rf build
# 假设有一个setup.py文件在当前目录下，执行它来构建扩展  
python setup.py build_ext --inplace
#清除中间文件
rm -f poly_cpu.cpp
# 递归删除build目录及其所有内容  
rm -rf build