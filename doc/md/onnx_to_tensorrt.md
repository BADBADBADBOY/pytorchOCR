### onnx转tensorrt
本项目使用tensorrt版本：TensorRT-7.0.0.11



#### 参数解释
pytorch_to_onnx.py 文件参数
|参数|含义|
|-|-|
|config|算法的配置文件|
|model_path|训练好的模型文件|
|img_path|测试的图片|
|save_path|onnx保存文件|
|batch_size|设置测试batch|
|max_size|设置最长边|
|algorithm|算法名称|
|add_padding|是否将短边padding到和长边一样|



onnx_to_tensorrt.py 文件参数
|参数|含义|
|-|-|
|onnx_path|生成的onnx文件|
|trt_engine_path|保存的engine文件路径|
|img_path|测试的图片|
|batch_size|设置测试batch|
|max_size|设置最长边|
|algorithm|算法名称|
|add_padding|是否将短边padding到和长边一样|

#### 单张调用
1. 生成onnx文件


- DB算法调用
	```
	python3 ./script/pytorch_to_onnx.py --config ./config/det_DB_mobilev3.yaml --model_path ./checkpoint/DB_best.pth.tar --img_path /src/notebooks/detect_text/icdar2015/ch4_test_images/img_10.jpg --save_path ./onnx/DB.onnx --batch_size 1 --max_size 1536 --algorithm DB --add_padding
	```
2. simple onnx文件

	```
	sh  onnx-simple.sh  DB.onnx  DB-simple.onnx
	```

3. 生成tensorrt engine
- DB算法调用
	```
	CUDA_VISIBLE_DEVICES=2 python3 ./script/onnx_to_tensorrt.py --onnx_path ./onnx/DB-simple.onnx --trt_engine_path ./onnx/DB.engine --img_path /src/notebooks/detect_text/icdar2015/ch4_test_images/img_10.jpg --batch_size 1 --algorithm DB --max_size 1536 --add_padding
	```

4. infer 调用

```
python3 ./tools/det_infer.py --config ./config/det_DB_mobilev3.yaml  --img_path /src/notebooks/detect_text/icdar2015/ch4_test_images  --result_save_path ./result --trt_path ./onnx/DB.engine --batch_size 1 --max_size 1536 --add_padding
```

#### batch 调用

操作同上，和单张调用一样，只是要把batch_size设置大于1 

- 提示：其余算法类似
