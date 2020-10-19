import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  
import onnx
import cv2
import numpy as np
import time
import argparse
import math

def resize_image(img,algorithm,side_len=1536,add_padding=True):
    
    if algorithm == 'CRNN':
        img = transform_img_shape(img,[32,crnn_max_length])
        return img
    
    if(algorithm=='SAST'):
        stride = 128
    else:
        stride = 32
    height, width, _ = img.shape
    flag = None
    if height > width:
        flag = True
        new_height = side_len
        new_width = int(math.ceil(new_height / height * width / stride) * stride)
    else:
        flag = False
        new_width = side_len
        new_height = int(math.ceil(new_width / width * height / stride) * stride)
    resized_img = cv2.resize(img, (new_width, new_height))
    if add_padding is True:
        if flag:
            padded_image = cv2.copyMakeBorder(resized_img, 0, 0,
                          0, side_len-new_width, cv2.BORDER_CONSTANT, value=(0,0,0))
        else:
            padded_image = cv2.copyMakeBorder(resized_img, 0, side_len-new_height,
                          0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
    else:
        return resized_img
    return padded_image


def build_engine(onnx_file_path,engine_file_path,batch_size=1,mode='fp16'):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING) 
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, load it instead of building a new one.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
        
    
    EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = int(batch_size)
        builder.fp16_mode = True if mode == 'fp16' else False
        builder.int8_mode = True if mode == 'int8' else False
        builder.strict_type_constraints = True
        
        builder.max_workspace_size = 1 << 32  # 1GB:30

        with open(onnx_file_path, 'rb') as model:
            parser.parse(model.read())

        print('network layers is',len(network))  # Printed output == 0. Something is wrong. 
        last_layer = network.get_layer(network.num_layers - 1)
        # Check if last layer recognizes it's output
        if not last_layer.get_output(0):
            # If not, then mark the output using TensorRT API
            network.mark_output(last_layer.get_output(0))
        engine = builder.build_cuda_engine(network)
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
    return engine


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(binding)), dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):

    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

def get_img(args):
    img = cv2.imread(args.img_path)
    img = resize_image(img,args.algorithm,side_len=args.max_size,add_padding=args.add_padding)
    print(img.shape)
    cv2.imwrite('./onnx/'+args.algorithm+'_ori_img.jpg',img)
    image = img / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

def gen_trt_engine(args):
    print("start check onnx file")
    test = onnx.load(args.onnx_path)
    onnx.checker.check_model(test)
    print("check onnx file Passed")
    print("build trt engine......")
    engine = build_engine(args.onnx_path,args.trt_engine_path,batch_size=int(args.batch_size))
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    context = engine.create_execution_context()
    print("build trt engine done")
    
    img1 = get_img(args)
    h,w = img1.shape[2:4]
    img_numpy = np.array([img1,img1])

    inputs[0].host = img_numpy

    s = time.time()
    output = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream,batch_size=int(args.batch_size))
    print(time.time()-s)
    if args.algorithm=='DB':
        out = output[0].reshape(int(args.batch_size),h,w)
        
        cv2.imwrite('./onnx/'+args.algorithm+'_trt_img1.jpg',out[0]*255)
        cv2.imwrite('./onnx/'+args.algorithm+'_trt_img2.jpg',out[1]*255)
    elif args.algorithm=='PAN':
        out = output[0].reshape(int(args.batch_size),6,h,w)
        cv2.imwrite('./onnx/'+args.algorithm+'_trt_img_text.jpg',out[1,0]*255)
        cv2.imwrite('./onnx/'+args.algorithm+'_trt_img_kernel.jpg',out[1,1]*255)
    elif args.algorithm=='PSE':
        out = output[0].reshape(int(args.batch_size),7,h,w)
        cv2.imwrite('./onnx/'+args.algorithm+'_trt_img_text.jpg',out[0,0]*255)
        cv2.imwrite('./onnx/'+args.algorithm+'_trt_img_kernel.jpg',out[0,6]*255)
    elif args.algorithm=='SAST':
        out = output[0].reshape(int(args.batch_size),h//4,w//4)
        out = out[0]
        cv2.imwrite('./onnx/'+args.algorithm+'_trt_img.jpg',out*255)
    else:
        print('not support this algorithm!!')



if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--onnx_path', help='config file path')
    parser.add_argument('--trt_engine_path', nargs='?', type=str, default=None)
    parser.add_argument('--img_path', nargs='?', type=str, default=None)
    parser.add_argument('--batch_size', nargs='?', type=str, default=None)
    parser.add_argument('--algorithm', nargs='?', type=str, default=None)
    parser.add_argument('--max_size', nargs='?', type=int, default=None)
    parser.add_argument('--add_padding', action='store_true', default=False)
    args = parser.parse_args()
    gen_trt_engine(args)