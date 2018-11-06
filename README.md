# Joint Video Stitching
An implementation of [Joint Video Stitching and Stabilization From Moving Cameras](https://ieeexplore.ieee.org/document/7563334) `[Heng Guo, Shuaicheng Liu, Tong He, Shuyuan Zhu,
Bing Zeng and Moncef Gabbouj]`

## Getting started
- python3.6
    ```
    pip install -r requirements.txt
    ```

- opencv4  [@Wei Wen](http://www.pittnuts.com/2015/12/build-opencv-in-ubutune/)

    which includes non-free modules such as xfeatures2d 
    
    ```
    mkdir ./resources/libs/opencv_build && cd ./resources/libs/opencv_build
    cmake ../opencv -DCMAKE_BUILD_TYPE=RELEASE \
     -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
     -DWITH_TBB=ON \
     -DBUILD_NEW_PYTHON_SUPPORT=ON \
     -DINSTALL_C_EXAMPLES=ON \
     -DINSTALL_PYTHON_EXAMPLES=ON \
     -DBUILD_EXAMPLES=ON \
     -DWITH_CUDA=OFF \
     -DBUILD_TIFF=ON \
     -DCMAKE_INSTALL_PREFIX=$(python -c "import sys; print(sys.prefix)") \
     -DPYTHON_EXECUTABLE=$(which python) \
     -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
     -DPYTHON_PACKAGES_PATH=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
     -DOPENCV_ENABLE_NONFREE=ON
    make -j5 # choose the num depends on your cpu 
    make install
    ```
    
```
python run.py
```