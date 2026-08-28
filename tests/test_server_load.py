# -*- coding: utf-8 -*-
"""模拟 app/server.py 的模型加载方式，验证修复"""
import sys, os
ROOT = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, ROOT)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from core_modules.face_recognition.face_recognition import FaceRecognitionSystem

face_recognition_model = None


def init():
    global face_recognition_model
    device = 'cuda:0'  # CUDA_VISIBLE_DEVICES=1 后可见设备内编号
    fr_device = 'tensorrt' if device.startswith('cuda') else device
    face_recognition_model = FaceRecognitionSystem(
        os.path.join(ROOT, 'weights/scrfd/scrfd_10g_bnkps.onnx'),
        os.path.join(ROOT, 'weights/arcface/Glint100.onnx'), device=fr_device)


init()
d = face_recognition_model.detector.session.get_providers()[0]
e = face_recognition_model.extractor.session.get_providers()[0]
print(f'SERVER_PATTERN_OK det={d} ext={e}')
assert d == 'TensorrtExecutionProvider' and e == 'TensorrtExecutionProvider'
print('ALL TRT - PASS')
