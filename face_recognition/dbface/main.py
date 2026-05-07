import dbface.common as common
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
from dbface.model.DBFace import DBFace
from config.base import TARGET_SIZE, FACE_THRESH, CUDA_DEVICE

# device = torch.device(f"cuda:{CUDA_DEVICE}" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# HAS_CUDA =""if torch.cuda.is_available()
# print(f"HAS_CUDA = {HAS_CUDA}")


def nms(objs, iou=0.5):

    if objs is None or len(objs) <= 1:
        return objs

    objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
    keep = []
    flags = [0] * len(objs)
    for index, obj in enumerate(objs):
        if flags[index] != 0:
            continue

        keep.append(obj)
        for j in range(index + 1, len(objs)):
            if flags[j] == 0 and obj.iou(objs[j]) > iou:
                flags[j] = 1

    return keep


def detect(model, image, threshold=FACE_THRESH, nms_iou=0.5):

    mean = [0.408, 0.447, 0.47]
    std = [0.289, 0.274, 0.278]

    image = common.pad(image)
    image = ((image / 255.0 - mean) / std).astype(np.float32)
    image = image.transpose(2, 0, 1)

    torch_image = torch.from_numpy(image)[None]
    # if HAS_CUDA:
        # torch_image = torch_image.cuda("cuda:" + str(CUDA_DEVICE))
    # torch_image = torch_image.to(device)
    torch_image = torch_image.to(device)

    hm, box, landmark = model(torch_image)
    hm_pool = F.max_pool2d(hm, 3, 1, 1)

    # 对提取不出1000个点的过小的人脸图片进行了过滤
    if hm.shape[2] * hm.shape[3] > 1000:

        scores, indices = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
        hm_height, hm_width = hm.shape[2:]

        scores = scores.squeeze()
        indices = indices.squeeze()
        ys = list((indices / hm_width).int().data.numpy())
        xs = list((indices % hm_width).int().data.numpy())
        scores = list(scores.data.numpy())
        box = box.cpu().squeeze().data.numpy()
        landmark = landmark.cpu().squeeze().data.numpy()

        stride = 4
        objs = []
        for cx, cy, score in zip(xs, ys, scores):
            if score < threshold:
                break

            x, y, r, b = box[:, cy, cx]
            xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
            x5y5 = landmark[:, cy, cx]
            x5y5 = (common.exp(x5y5 * 4) + ([cx] * 5 + [cy] * 5)) * stride
            box_landmark = list(zip(x5y5[:5], x5y5[5:]))
            objs.append(common.BBox(0, xyrb=xyrb, score=score, landmark=box_landmark))
        return nms(objs, iou=nms_iou)
    else:
        return []


def resize_img_keep_ratio(img_name, target_size):
    old_size = img_name.shape[0:2]

    # 找到 目标尺寸/图片尺寸 的小值，也就是相对长边
    ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))
    # 计算按原比例缩放后的目标尺寸
    new_size = tuple([int(i * ratio) for i in old_size])
    img = cv2.resize(img_name, (new_size[1], new_size[0]))

    pad_w = target_size[1] - new_size[1]
    pad_h = target_size[0] - new_size[0]
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    img_new = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (255, 255, 255)
    )
    return img_new


def detect_image(model, image, pixel_threshold):
    # image = common.imread(file)      # 修改为cv读取后的图片传入，不需要再进行读取
    image_h, image_w = image.shape[0:2]
    """    
    # 指定目标最长边大小，用于控制人脸图片的大小和显存占用
    # 如果图像尺寸大于目标阈值，则缩小图像最大边到阈值大小
    # 图片太大的情况，需要先resize再进网络，防止炸显存
    """
    ratio = 1
    if max(image_h, image_w) > TARGET_SIZE:

        old_size = image.shape[0:2]
        # 找出相对长边
        ratio = min(float(TARGET_SIZE) / (old_size[i]) for i in range(len(old_size)))
        # 计算按原比例缩放后的目标尺寸
        new_size = tuple([int(i * ratio) for i in old_size])
        image = cv2.resize(image, (new_size[1], new_size[0]))

    objs = detect(model, image)

    # 修改为只画出检测面积最大的
    area_list = []
    for obj in objs:
        area = obj.width * obj.height
        area_list.append(area)
    # 在面积的列表中找出像素大于指定数量的值和索引
    filter_faces = [
        index for index, value in enumerate(area_list) if value > pixel_threshold
    ]  # 这个阈值用来筛选掉不需要的较小的人脸框
    face_rects = []
    for index_num in filter_faces:
        obj = objs[index_num]

        # print(obj)
        face_rect = (obj.x, obj.y, obj.width, obj.height)
        # print(face_rect)
        face_rects.append(face_rect)

    return face_rects, image, ratio
