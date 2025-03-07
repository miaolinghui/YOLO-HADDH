<<<<<<< HEAD
# YOLO-HADDH
A Novel Lightweight Network based on Hybrid Architecture and Deformable Decoupled Head
=======
The YOLO_HADDH is a target detection model we proposed based on a lightweight hybrid architecture and an efficient deformable decoupled head. It has been validated on the VOC PASCAL (2007+2012) and MS COCO (2014) datasets. With little change in the number of parameters and computational complexity, its detection accuracy in all aspects is significantly higher than that of the baseline YOLOv5n model.

On the PASCAL VOC dataset, we used the pre-trained weights [YOLOv5n](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt). The model was trained for 500 epochs on an Ubuntu 20.04 system equipped with a 3090ti GPU.

On the MS COCO dataset, no pre-trained weights were used. The model was trained for 300 epochs on an Ubuntu 20.04 system with dual 4090 GPUs.

Its performance metrics are as follows:

| Datasets   | size(pixels) | mAPval50-95 | mAPval50 | mAPval75 | mAPvalsmall | mAPvalmiddle | mAPvallarge | GFLOPs | params(M) | Lantency(ms) |
| :--------- | :----------- | :---------- | :------- | :------- | :---------- | :----------- | :---------- | :----- | :-------- | :----------- |
| PASCAL VOC | 640          | 56.7        | 79.1     | 62.4     | 24.2        | 39.5         | 63.0        | 6.33   | 3.48      | 4.41         |
| MS COCO    | 640          | 33.5        | 51.5     | 35.4     | 16.5        | 37.2         | 45.2        | 6.59   | 3.57      | 4.33         |

## <div align="center"></div>

For complete documentation on training, testing, and deployment, please refer to the quickstart examples below.

Clone repo，and ensure that [**Python>=3.8.0**](https://www.python.org/) is installed in your environment. Additionally, install the dependencies listed in [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) ，and ensure that you have  [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/) installed。

```
git clone https://github.com/lvyongshjd/YOLO-HADDH
cd YOLO-HADDH

# Install dependencies required by YOLOv5
pip install -r requirements.txt
pip install einops

# Compile DCNv4: Navigate to the directory containing DCNv4_op and execute:
python setup.py build install
```

Note that the VOC PASCAL and MS COCO datasets should be placed in the `datasets` directory at the same level as the YOLO-HADDH directory.

```
#Since the DCNv4 module is not applicable to images that are not 640×640, the matrix training and rect parameters are defined as False. The model YOLO-HADDH.yaml integrates all the improved and innovative models of this project.
#Navigate to the YOLO-HADDH directory and run the following code to start training. If you have insufficient GPU memory, you can adjust the batch size. Due to the risk of gradient explosion, the initial batch size should not exceed 8 to prevent the loss from exceeding the limit and becoming NaN.
python train.py
```

验证测试模型

```
#After training, an exp folder containing all the training result files will be created in the runs/train directory under the YOLO-HADDH directory. The best.pt file in the weights folder can be used to call the validation script:
python val.py
#The rect parameter should still be consistent with the original training and kept as False. Enabling the save_json parameter allows automatic retrieval of COCO metrics.
#To measure GFLOPs, you can use the GFLOPs.py script in the YOLO-HADDH folder:
python GFLOPs
#To measure FPS, you can use the test.py script in the YOLO-HADDH folder:
python test.py
```

利用训练好的模型权重检测图片

```bash
#Utilizing the trained weights, since the DCNv4 module can only perform forward propagation on images of size 640×640, we need to pad the images to 640×640.
python pad.py  # Read image addresses and generate padded images in the corresponding directory.
python detect.py  # Generate images with detection results in the corresponding folder.
```

>>>>>>> 79a38cc (提交文件)
