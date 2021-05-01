### pytorch 转onnx

####  1. 运行根目录to_onnx.sh文件:

```
sh to_onnx.sh
```
里面有四个参数，需要对应修改，参数解释如下：
|参数|解释|
|-|-|
|config|对应算法的config文件|
|model_path|对应算法的模型文件|
|img_path|测试图片|
|save_path|保存onnx文件|
- 提示：这里onnx文件建议生成在项目中onnx文件夹下
#### 2. 使用onnx文件夹下的 onnx-simple.sh对生成的onnx文件进行精简，运行：

```
sh onnx-simple.sh 生成的onnx文件 精简后的onnx文件
```
例如：

```
sh  onnx-simple.sh  DBnet.onnx  DBnet-simple.onnx
```
#### 3. onnx调用
运行：

```
python3 ./tools/det_infer.py --config ./config/det_DB_mobilev3.yaml --model_path ./checkpoint/ag_DB_bb_mobilenet_v3_small_he_DB_Head_bs_16_ep_400/DB_64.pth.tar --img_path /src/notebooks/detect_text/icdar2015/ch4_test_images/img_10.jpg --result_save_path ./result --onnx_path ./onnx/DBnet-simple.onnx
```
- 提示：这里如果加上--onnx_path就是onnx调用，否则是pytorch调用


