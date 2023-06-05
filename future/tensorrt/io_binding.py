import torch
# import nvidia
import onnxruntime as ort

import os
import time



# 延迟加载
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
new_path = r'E:\GitHub\image-deduplicate-cluster-webui\venv\Lib\site-packages\torch\lib'
os.environ['PATH'] += os.pathsep + new_path


# 打印 ONNX Runtime 版本信息
print("ONNX Runtime version:", ort.__version__)

# 检查是否可用 GPU
if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is NOT available")

# EP_list = ort.get_available_providers()
# EP_list = ['TensorrtExecutionProvider']
# EP_list = ['CUDAExecutionProvider']
# EP_list = ['CPUExecutionProvider']
# EP_list = ['TensorrtExecutionProvider']
# print(EP_list)

model_path = r"wd14_tagger_model\model.onnx"

os.makedirs(os.path.join(os.path.dirname(model_path), "trt_engine_cache"), exist_ok=True)


trt_ep_options = {
    "trt_timing_cache_enable": True,
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path":os.path.join(os.path.basename(model_path), "trt_engine_cache"),
}

input_arr = torch.randn( (1, 448, 448, 3) ).cuda()

def run(provider):

    
    main_start = time.time()

    start_time = time.time()
    # initialize the model.onnx
    sess = ort.InferenceSession(model_path, providers=provider)

    print(provider)
    print("模型载入耗时：", time.time() - start_time)

    # get the outputs metadata as a list of :class:`onnxruntime.NodeArg`
    output_name = sess.get_outputs()[0].name

    # io绑定
    io_binding = sess.io_binding()

    import numpy as np
    # 获取输出
    start = time.time()
    for i in range(20):
        # input_arr = np.random.randint(0, 225, size=(1, 448, 448, 3)).astype(np.float32)
        io_binding.bind_input(name=sess.get_inputs()[0].name,
                            device_type='cuda',
                            device_id=0,
                            element_type=np.float32,
                            shape=tuple(input_arr.shape),
                            buffer_ptr=input_arr.data_ptr()
        )
        io_binding.bind_output(name=sess.get_outputs()[0].name)          
        sess.run_with_iobinding(io_binding)
        ort_outs = io_binding.copy_outputs_to_cpu()[0]

        # ort_inputs = {sess.get_inputs()[0].name: input_arr,}
        # ort_outs = sess.run(None, ort_inputs)

    print("推理耗时:", time.time() - start)

    print("总耗时：", time.time() - main_start)

    return ort_outs

tensorrt_ort_outs = run([("TensorrtExecutionProvider", trt_ep_options)])
cuda_ort_outs = run(['CUDAExecutionProvider'])

import numpy as np
print( np.max(np.abs(tensorrt_ort_outs - cuda_ort_outs)) )