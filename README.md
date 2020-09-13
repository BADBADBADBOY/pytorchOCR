## 基于pytorch的OCR库


目前已完成:

- [x] DBnet [论文链接](https://arxiv.org/abs/1911.08947)
- [x] PSEnet [论文链接](https://arxiv.org/abs/1903.12473)
- [x] PANnet [论文链接](https://arxiv.org/pdf/1908.05900.pdf)
- [x] SASTnet [论文链接](https://arxiv.org/abs/1908.05498)
- [x] CRNN [论文链接](https://arxiv.org/abs/1507.05717)

接下来计划：
- [ ] 训练通用化ocr模型
- [ ] 模型转onnx及调用测试
- [ ] 模型压缩（剪枝和量化）
- [ ] 模型蒸馏
- [ ] tensorrt部署
- [ ] 手机端部署

### 检测模型效果(实验中)

训练只在ICDAR2015文本检测公开数据集上，算法效果如下：
|模型|骨干网络|precision|recall|Hmean|下载链接|
|-|-|-|-|-|-|
|DB|ResNet50_7*7|85.88%|79.10%|82.35%|[下载链接](https://pan.baidu.com/s/1zONYFPsS3szaf5BHeQh5ZA)(code:fxw6)|
|DB|ResNet50_3*3|86.51%|80.59%|83.44%|[下载链接](https://pan.baidu.com/s/1zONYFPsS3szaf5BHeQh5ZA)(code:fxw6)|
|DB|MobileNetV3|82.89%|75.83%|79.20%|[下载链接](https://pan.baidu.com/s/1zONYFPsS3szaf5BHeQh5ZA)(code:fxw6)|
|SAST|ResNet50_7*7|85.72%|78.38%|81.89%|[下载链接](https://pan.baidu.com/s/1zONYFPsS3szaf5BHeQh5ZA)(code:fxw6)|
|SAST|ResNet50_3*3|86.67%|76.74%|81.40%|[下载链接](https://pan.baidu.com/s/1zONYFPsS3szaf5BHeQh5ZA)(code:fxw6)|
|PSE|ResNet50_7*7|0%|0%|0%|[下载链接](https://pan.baidu.com/s/1zONYFPsS3szaf5BHeQh5ZA)(code:fxw6)|
|PSE|ResNet50_3*3|0%|0%|0%|[下载链接](https://pan.baidu.com/s/1zONYFPsS3szaf5BHeQh5ZA)(code:fxw6)|
|PAN|ResNet18_7*7|81.80%|77.08%|79.37%|[下载链接](https://pan.baidu.com/s/1zONYFPsS3szaf5BHeQh5ZA)(code:fxw6)|
|PAN|ResNet18_3*3|83.78%|75.15%|79.23%|[下载链接](https://pan.baidu.com/s/1zONYFPsS3szaf5BHeQh5ZA)(code:fxw6)|


### 环境安装和预训练模型
1. 编译c++后处理文件

```
sh make.sh
```
2. 下载预训练模型
预训练模型地址：[下载链接](https://pan.baidu.com/s/1zONYFPsS3szaf5BHeQh5ZA)(code:fxw6)

3. 下载icdar2015测试模型(不做测试可跳过这一步)
测试模型地址：[下载链接](https://pan.baidu.com/s/1zONYFPsS3szaf5BHeQh5ZA)(code:fxw6)

4. 将下载下来的模型文件夹替换项目中同名文件夹

### 文本检测
#### 数据准备
照着icdar2015的格式, x1,y1,x2,y2,x3,y3,x4,y4,label，其中不参与训练文本（例如模糊文本），label设置为###，其他设置text.
```

image
│   .jpg
│   .jpg   
│		...
label
│   .txt
│   .txt
|		...
```
需要一个train_list.txt , 格式：图片绝对路径+\t+label。 具体可参照项目中data/example中例子。如果训练过程中需要做验证，需要制作相同的数据格式有一个test_list.txt。

#### 训练模型
为了方便实验，暂时没写成args输入的形式。
1. 修改./config中对应算法的yaml中参数，基本上只需修改数据路径即可。
2. 在./tools/det_train.py最下面打开不同的config中的yaml对应不同的算法
3. 运行下面命令

```
python3 ./tools/det_train.py
```
#### 测试模型
1. 修改infer.sh中的参数
2. 运行下面命令

```
sh infer.sh
```



### 文本识别
#### 数据准备

```
image
│   .jpg
│   .jpg   
│		...

```
需要一个train_list.txt , 格式：图片绝对路径+\t+label。 具体可参照项目中data/example中例子。如果训练过程中需要做验证，需要制作相同的数据格式有一个test_list.txt。

#### 训练模型
1. 修改./config中对应算法的yaml中参数，基本上只需修改数据路径即可。
2. 在./tools/rec_train.py最下面打开不同的config中的yaml对应不同的算法
3. 运行下面命令

```
python3 ./tools/rec_train.py
```
#### 测试模型
1. 运行下面命令

```
python3 ./tools/rec_infer.py
```

### 参考
- https://github.com/PaddlePaddle/PaddleOCR
- https://github.com/whai362/PSENet
- https://github.com/whai362/pan_pp.pytorch
- https://github.com/WenmuZhou/PAN.pytorch
- https://github.com/xiaolai-sqlai/mobilenetv3
- https://github.com/BADBADBADBOY/DBnet-lite.pytorch
- https://github.com/BADBADBADBOY/Psenet_v2
- https://github.com/BADBADBADBOY/pse-lite.pytorch