import os
import torch
import numpy as np
from facenet.detect import Detect
import torch.nn.functional as F
import cv2


class FaceImages:

    def __init__(self):
        # 用于提取特征向量的神经网络，可更换
        self.detect = Detect()

    # 提取特征向量
    def extract_feature(self, image_path):
        image = cv2.imread(image_path)
        self.feature = self.detect(image)
        return self.feature

    # 特征对比
    def feature_compare(self, feature):
        result = {}
        for name in os.listdir(self.feature_path):
            path = os.path.join(self.feature_path, name)
            features = torch.load(path)
            res = F.cosine_similarity(features, feature)
            result[name] = res
        return result

    # 遍历文件夹中的所有图片进行向量提取
    def extract_face_image_path(self, compare_image_path):
        feature_list = []
        for compare_images in os.listdir(compare_image_path):
            compare_image = os.path.join(compare_image_path, compare_images)

            image = cv2.imread(compare_image)
            # 提取待检测的图像, tensor
            self.feature = self.detect(image)
            # feature = self.feature.tolist()[0]
            feature_info = {
                "image_path": compare_images,
                "feature": self.feature,
            }
            feature_list.append(feature_info)
        # 返回特征向量以及其对应的文件路径
        return feature_list

    # 图片特征向量提取
    def extract_face_image(self, compare_image_path):
        image = cv2.imread(compare_image_path)
        # 提取待检测的图像,tensor
        feature = self.detect(image)
        return feature, compare_image_path

    # 两张图片的特征向量比较
    def compare_face_image(self, image_1, image_2):
        feature_1, image_1 = self.extract_face_image(image_1)
        feature_2, image_2 = self.extract_face_image(image_2)
        similar = F.cosine_similarity(feature_1, feature_2)
        infomation = f"{os.path.basename(image_1)}与{os.path.basename(image_2)}相似度为{similar.item()}"
        return infomation

    # 一张图片与文件夹中所有图片的特征向量比较
    def compare_image2image_path(self, image, image_path):
        feature_1, image_1 = self.extract_face_image(image)
        feature_list = self.extract_face_image_path(image_path)
        infomation_list = []
        for feature in feature_list:
            feature_2 = feature["feature"]
            image_2 = feature["image_path"]
            similar = F.cosine_similarity(feature_1, feature_2)
            infomation = f"{os.path.basename(image_1)}与{os.path.basename(image_2)}相似度为{similar.item()}"
            infomation_list.append(infomation)
        return infomation_list


if __name__ == "__main__":
    face_images = FaceImages()

    # # 查看特征向量的值和维度
    # feature, image_path = face_images.extract_face_image(
    #     ""
    # )
    # print(feature.tolist()[0])
    # print(len(feature.tolist()[0]))

    # # 对比两张图片
    # similar_info = face_images.compare_face_image(
    #     "",
    #     "",
    # )

    # 对比一张图片与文件夹中的所有图片
    # similar_info_list = face_images.compare_image2image_path(
    #     "",
    #     "",
    # )

    # 提取一张图片的人脸向量
    image_path = ""
    featrue = face_images.extract_feature(image_path)
    print(featrue.tolist()[0])
