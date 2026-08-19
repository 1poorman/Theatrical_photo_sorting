# -*- coding: utf-8 -*-
"""
SCRFD 人脸检测器（ONNX 推理）
参考 insightface 官方实现 python-package/insightface/model_zoo/scrfd.py
使用 weights/scrfd/scrfd_10g_bnkps.onnx（9 输出：3 cls + 3 bbox + 3 kps）
"""
import os
import os.path as osp
import numpy as np
import cv2
import onnxruntime


class FaceBBox:
    """轻量人脸框（替代原 dbface.common.BBox，供兼容接口使用）"""
    def __init__(self, x, y, r, b, score=0.0, landmark=None):
        self.x = x
        self.y = y
        self.r = r
        self.b = b
        self.score = score
        self.landmark = landmark

    @property
    def width(self):
        return self.r - self.x + 1

    @property
    def height(self):
        return self.b - self.y + 1


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.（官方实现）"""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clip(min=0, max=max_shape[1])
        y1 = y1.clip(min=0, max=max_shape[0])
        x2 = x2.clip(min=0, max=max_shape[1])
        y2 = y2.clip(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to keypoints.（官方实现）"""
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clip(min=0, max=max_shape[1])
            py = py.clip(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class SCRFDDetector:
    """SCRFD 人脸检测器（onnxruntime）"""

    def __init__(self, model_path=None, device=None, det_thresh=0.5, nms_thresh=0.4,
                 input_size=(640, 640)):
        """
        Args:
            model_path: scrfd_10g_bnkps.onnx 路径
            device: 'cpu' 或 'cuda'（onnxruntime GPU 需 onnxruntime-gpu）
            det_thresh: 检测置信度阈值
            nms_thresh: NMS IoU 阈值
            input_size: 检测输入尺寸 (width, height)，onnx 固定为 640x640
        """
        self.model_file = model_path
        self.det_thresh = det_thresh
        self.nms_thresh = nms_thresh
        self.input_size = tuple(input_size)
        self.use_kps = False
        self.center_cache = {}
        self._num_anchors = 1
        self._feat_stride_fpn = [8, 16, 32]

        if model_path is None or not osp.exists(model_path):
            raise FileNotFoundError(f"SCRFD model not found: {model_path}")

        providers = self._select_providers(device)
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        self._init_vars()
        print(f"SCRFD loaded from {model_path} (providers={providers})")

    def _select_providers(self, device):
        available = onnxruntime.get_available_providers()
        if device is None or device.startswith('cuda'):
            if 'CUDAExecutionProvider' in available:
                return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        return ['CPUExecutionProvider']

    def _init_vars(self):
        """从 onnx 图读取输入输出信息（官方实现）"""
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        if isinstance(input_shape[2], str):
            self.input_size = None
        else:
            self.input_size = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape
        self.input_name = input_cfg.name

        outputs = self.session.get_outputs()
        self.output_names = [o.name for o in outputs]
        self.input_mean = 127.5
        self.input_std = 128.0

        num_out = len(outputs)
        self.fmc = 3
        if num_out == 9:  # scrfd_10g_bnkps: 3 cls + 3 bbox + 3 kps
            self._num_anchors = 2
            self.use_kps = True
        elif num_out == 6:
            self._num_anchors = 2
        elif num_out == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self.use_kps = True
        elif num_out == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
        print(f"SCRFD: outputs={num_out}, fmc={self.fmc}, use_kps={self.use_kps}, "
              f"num_anchors={self._num_anchors}")

    def prepare(self, ctx_id, **kwargs):
        """兼容官方接口"""
        if ctx_id < 0:
            self.session.set_providers(['CPUExecutionProvider'])
        self.nms_thresh = kwargs.get('nms_thresh', self.nms_thresh)
        self.det_thresh = kwargs.get('det_thresh', self.det_thresh)

    def forward(self, img, threshold):
        """网络前向 + 解码（官方实现）"""
        scores_list = []
        bboxes_list = []
        kpss_list = []

        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(
            img, 1.0 / self.input_std, input_size,
            (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = net_outs[idx][0]
            bbox_preds = net_outs[idx + fmc][0]
            bbox_preds = bbox_preds * stride
            if self.use_kps:
                kps_preds = net_outs[idx + fmc * 2][0] * stride

            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        return scores_list, bboxes_list, kpss_list

    def detect(self, img, input_size=None, max_num=0, metric='default'):
        """检测人脸（保持长宽比缩放 + pad，官方实现）
        Returns:
            det: (N, 5) [x1, y1, x2, y2, score]
            kpss: (N, 5, 2) 关键点（若 use_kps）
        """
        if input_size is None:
            input_size = self.input_size
        assert input_size is not None
        input_size = tuple(input_size)

        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self.forward(det_img, self.det_thresh)

        if len(scores_list) == 0 or sum(s.shape[0] for s in scores_list) == 0:
            return np.empty((0, 5), dtype=np.float32), \
                   (np.empty((0, 5, 2), dtype=np.float32) if self.use_kps else None)

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None

        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
            if metric == 'max':
                values = area
            else:
                values = area - offset_dist_squared * 2.0
            bindex = np.argsort(values)[::-1]
            bindex = bindex[0:max_num]
            det = det[bindex, :]
            if kpss is not None:
                kpss = kpss[bindex, :]
        return det, kpss

    def nms(self, dets):
        """标准 NMS（官方实现）"""
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

    # ---- 兼容原有检测器接口 ----
    def _detect_single(self, image, threshold=None, nms_iou=None):
        """兼容接口：返回 FaceBBox 列表（带 landmark）"""
        det, kpss = self.detect(image)
        objs = []
        if det.shape[0] == 0:
            return objs
        for i in range(det.shape[0]):
            x1, y1, x2, y2, score = det[i]
            landmark = None
            if kpss is not None and kpss.shape[0] > i:
                landmark = [(kp[0], kp[1]) for kp in kpss[i]]
            objs.append(FaceBBox(x1, y1, x2, y2, score=float(score), landmark=landmark))
        return objs

    def detect_image(self, image, pixel_threshold=0):
        """兼容接口：返回 (face_rects, processed_image, ratio)"""
        image_h, image_w = image.shape[0:2]
        ratio = 1
        TARGET_SIZE = 1920
        if max(image_h, image_w) > TARGET_SIZE:
            old_size = image.shape[0:2]
            ratio = min(float(TARGET_SIZE) / old_size[i] for i in range(len(old_size)))
            new_size = tuple([int(i * ratio) for i in old_size])
            image = cv2.resize(image, (new_size[1], new_size[0]))

        objs = self._detect_single(image)
        face_rects = []
        for obj in objs:
            if obj.width * obj.height > pixel_threshold:
                face_rects.append((obj.x, obj.y, obj.width, obj.height))
        return face_rects, image, ratio


def get_scrfd(name, download=False, root='~/.insightface/models', **kwargs):
    if not download:
        assert os.path.exists(name)
        return SCRFDDetector(model_file=name, **kwargs)
    from .model_store import get_model_file
    _file = get_model_file("scrfd_%s" % name, root=root)
    return SCRFDDetector(model_file=_file, **kwargs)


if __name__ == '__main__':
    det = SCRFDDetector(
        model_path='./weights/scrfd/scrfd_10g_bnkps.onnx',
        device='cpu', det_thresh=0.3)
    img = cv2.imread('data/sample_images/4.jpg')
    import time
    t0 = time.time()
    bboxes, kpss = det.detect(img)
    print('time:', round(time.time() - t0, 3), 's')
    print('det:', bboxes.shape, 'kps:', kpss.shape if kpss is not None else None)
    if bboxes.shape[0] > 0:
        for i in range(min(3, bboxes.shape[0])):
            print(f'face {i}: {bboxes[i][:4].astype(int)} score={bboxes[i][4]:.3f}')
        print('kps[0]:', kpss[0])
