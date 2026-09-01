# -*- coding: utf-8 -*-
"""
TensorRT 加速支持模块
- 预加载 TensorRT 动态库（libnvinfer.so.10 等），使 onnxruntime 的
  TensorrtExecutionProvider 可用
- 提供统一的 provider 选择逻辑与引擎缓存配置（FP16 + timing cache）
- 支持直接使用原生 tensorrt Python API 构建引擎（可选路径）

用法（onnxruntime 路径，推荐，改动最小）：
    from trt_utils import create_ort_session
    session = create_ort_session(model_path, device='tensorrt', fp16=True)
"""
import os
import glob
import ctypes

# 默认 FP16 开关（RTX 3090 FP16 吞吐约为 FP32 两倍）
DEFAULT_FP16 = os.environ.get('FACE_TRT_FP16', '1') not in ('0', 'false', 'False')
# 引擎缓存目录（首次构建引擎较慢，之后直接加载缓存）
TRT_CACHE_DIR = os.environ.get(
    'FACE_TRT_CACHE',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trt_cache'))

_is_loaded = False


def preload_trt_libs():
    """预加载 TensorRT 动态库，解决 libnvinfer.so.10 找不到的问题。

    pip 安装的 tensorrt-cu12 库文件位于 site-packages/tensorrt_libs/，
    不在默认 ld 路径中，需显式 ctypes.CDLL 加载。
    """
    global _is_loaded
    if _is_loaded:
        return True
    import importlib.util
    for pkg in ('tensorrt_libs', 'tensorrt_cu12.libs', 'nvidia.tensorrt'):
        spec = importlib.util.find_spec(pkg)
        if spec is None or not spec.submodule_search_locations:
            continue
        lib_dir = list(spec.submodule_search_locations)[0]
        loaded_main = False
        # 按依赖顺序加载
        for pattern in ('libnvinfer_builder_resource.so*', 'libnvinfer.so*',
                        'libnvonnxparser.so*', 'libnvinfer_plugin.so*'):
            for lib in sorted(glob.glob(os.path.join(lib_dir, pattern))):
                try:
                    ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                    if 'libnvinfer.so' in os.path.basename(lib):
                        loaded_main = True
                        print(f"[trt_utils] preloaded {os.path.basename(lib)}")
                except OSError as e:
                    print(f"[trt_utils] skip {os.path.basename(lib)}: {e}")
        if loaded_main:
            _is_loaded = True
            return True
    return False


def get_trt_provider_options(cache_prefix, input_name=None, input_shape=None):
    """构造 TensorrtExecutionProvider 的 session options 参数

    Args:
        input_shape: 模型输入 shape（如 ArcFace 为 (1,3,112,112)）。
            提供时显式设置 profile 覆盖 batch 1..32，一个引擎服务所有 batch，
            避免运行中遇到新 batch 尺寸触发数十秒的引擎重建（多脸图片耗时暴增的根因）。
    """
    os.makedirs(TRT_CACHE_DIR, exist_ok=True)
    opts = {
        'device_id': 0,
        'trt_max_workspace_size': 4 << 30,          # 4GB
        'trt_fp16_enable': DEFAULT_FP16,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': TRT_CACHE_DIR,
        'trt_timing_cache_enable': True,
        'trt_timing_cache_path': TRT_CACHE_DIR,
        'trt_engine_cache_prefix': cache_prefix,
        'trt_force_sequential_engine_build': False,
    }
    if input_name and input_shape:
        # 仅当输入 shape 全为静态整数时设置显式 profile（min/opt/max 三者必须齐全，
        # 缺 opt 会被 ORT 整体作废并回退隐式 profile，导致运行中引擎重建）
        def _dims(s):
            return 'x'.join(str(d if d > 0 else 1) for d in s)
        dims = input_shape[1:]
        if all(isinstance(d, int) and d > 0 for d in dims):
            min_b, opt_b, max_b = 1, 8, 32
            shape_suffix = _dims(dims)
            opts['trt_profile_min_shapes'] = f"{input_name}:{min_b}x{shape_suffix}"
            opts['trt_profile_opt_shapes'] = f"{input_name}:{opt_b}x{shape_suffix}"
            opts['trt_profile_max_shapes'] = f"{input_name}:{max_b}x{shape_suffix}"
    return opts


def create_ort_session(model_path, device='cpu', fp16=None, log_severity=3):
    """创建 onnxruntime InferenceSession，统一封装 provider 选择。

    Args:
        model_path: onnx 模型路径
        device: 'cpu' / 'cuda' / 'tensorrt'
        fp16: 是否启用 FP16（仅 tensorrt 生效，None 则用 DEFAULT_FP16）
    Returns:
        onnxruntime.InferenceSession
    """
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = log_severity

    if device == 'tensorrt' or (isinstance(device, str) and device.startswith('tensorrt:')):
        preload_trt_libs()
        # 支持 'tensorrt:N' 语法指定 GPU 编号
        device_id = 0
        if isinstance(device, str) and ':' in device:
            try:
                device_id = int(device.split(':', 1)[1])
            except ValueError:
                pass
        prefix = os.path.splitext(os.path.basename(model_path))[0]
        po = get_trt_provider_options(cache_prefix=prefix)
        # 预读模型输入（静态 shape）设置 batch 1..32 的 profile，一个引擎覆盖所有 batch
        try:
            _probe = ort.InferenceSession(model_path, so, providers=['CPUExecutionProvider'])
            _inp = _probe.get_inputs()[0]
            po = get_trt_provider_options(
                cache_prefix=prefix, input_name=_inp.name, input_shape=tuple(_inp.shape))
        except Exception:
            pass
        po['device_id'] = device_id
        if fp16 is not None:
            po['trt_fp16_enable'] = fp16
        session = ort.InferenceSession(
            model_path, so,
            providers=[('TensorrtExecutionProvider', po),
                       'CUDAExecutionProvider', 'CPUExecutionProvider'])
        # TRT 不可用时 onnxruntime 会静默回退，这里显式校验
        actual = session.get_providers()[0]
        if actual != 'TensorrtExecutionProvider':
            print(f"[trt_utils] WARNING: TRT EP 不可用，回退到 {actual}")
    elif device == 'cuda':
        session = ort.InferenceSession(
            model_path, so,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    else:
        session = ort.InferenceSession(
            model_path, so, providers=['CPUExecutionProvider'])

    return session


if __name__ == '__main__':
    import onnxruntime as ort
    print('available providers:', ort.get_available_providers())
    ok = preload_trt_libs()
    print('trt libs preloaded:', ok)
