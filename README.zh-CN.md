YOLO_HADDH是我们提出的基于一种基于轻量化混合架构及高效可变形解耦头的目标检测模型，经过在VOC PASCAL（2007+2012）及MS COCO（2014）数据集上的验证，其在参数量及计算复杂度变化不大的情况下，各项检测精度均显著高于基线YOLOv5n模型的表现。

在PASCAL VOC数据集上采用预训练权重[YOLOv5n](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt)，训练 500 epochs，在搭配3090ti 显卡的ubuntu20.04系统上进行。

在MS COCO数据集上则不采用预训练权重，训练 300 epochs，在搭配4090双显卡的ubuntu20.04系统上进行。

其指标表现如下：

| Datasets   | size(pixels) | mAPval50-95 | mAPval50 | mAPval75 | mAPvalsmall | mAPvalmiddle | mAPvallarge | GFLOPs | params(M) | Lantency(ms) |
| :--------- | :----------- | :---------- | :------- | :------- | :---------- | :----------- | :---------- | :----- | :-------- | :----------- |
| PASCAL VOC | 640          | 56.7        | 79.1     | 62.4     | 24.2        | 39.5         | 63.0        | 6.33   | 3.48      | 4.41         |
| MS COCO    | 640          | 33.5        | 51.5     | 35.4     | 16.5        | 37.2         | 45.2        | 6.59   | 3.57      | 4.33         |

## <div align="center">文档</div>

有关训练、测试和部署的完整文档请参阅下面的快速入门示例。

克隆 repo，并要求在 [**Python>=3.8.0**](https://www.python.org/) 环境中安装 [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) ，且要求 [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/) 。

```
git clone https://github.com/lvyongshjd/YOLO-HADDH
cd YOLO-HADDH
#YOLOv5所需要的依赖项
pip install -r requirements.txt
pip install einops
#DCNv4编译，进入文件目录下找到DCNv4_op然后执行：
python setup.py build install
```

训练模型注意VOC PASCAL及MS COCO数据集要在YOLO-HADDH的同级目录底下的datasets里

```
#在执行的文件里由于DCNv4模块不适用与非640*640的图像所以，将矩阵训练及rect参数定义为Fasle，选择模型YOLO-HADDH.yaml集成了本项目所有的改进创新后的模型
#进入YOLO-HADDH目录下执行下列代码即可开始训练，显存不足可调节batchsize，起始batchsize由于梯度爆炸的原因不建议超过8，以防止loss超越限制NaN
python train.py
```

验证测试模型

```
#训练结束后会在YOLO-HADDH目录下的runs文件夹底下的train文件夹里形成exp文件包含所有的训练结果文件，其中weights中的best.pt可以用于调用来执行
python val.py
#其中参数rect仍然需要和原本的train一致保持Fasle，save_jason参数开启后可自动读取COCO指标
#测量GFLOPs可以采用YOLO-HADDH文件夹中的GFLOPs.py
python GFLOPs
#测量FPS可以采用YOLO-HADDH文件夹中的test.py
python test.py
```

利用训练好的模型权重检测图片

```
#利用训练得到的权重，由于DCNv4的模块只能针对640*640图像前向传播，所以我们需要将图片填充成640*640进行
python pad.py#读取图片地址在对应的文件中生成填充好的图片
python detect.py#在对应的文件夹里生成带有检测结果的图片
```

