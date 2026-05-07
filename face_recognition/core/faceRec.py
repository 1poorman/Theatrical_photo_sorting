import os
import base64
import cv2
import numpy as np
from PIL import Image
import shutil
from config.logConfig import *
from infra.bizException import *
from dbface.main import detect_image
from algo.dbface import dbface


def image_to_base64(image_path):
    with open(image_path, "rb") as file:
        with open("tests/1.txt", "wb") as output:
            base64_data = base64.b64encode(file.read())
            output.write(base64_data)
    return base64_data


def base64_to_cv(img_base64):
    try:
        img_decode = base64.b64decode(img_base64)
        img_array = np.fromstring(img_decode, np.uint8)
        img = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)
        return img
    except Exception as ex:
        logger.error("Base64图像转换异常, ex={}".format(ex))
        raise BizException("Base64图像转换异常")


def face_recognition_path(
    image_path,
    dbface,
    pixel_threshold,
    video_output_path=None,
    use_blur_detection=False,
):
    base_name = os.path.basename(image_path)
    output_path = os.path.dirname(image_path)
    image = cv2.imread(image_path)

    return face_recognition(
        image,
        dbface,
        pixel_threshold,
        output_path=output_path,
        base_name=base_name,
        image_path=image_path,
        video_output_path=video_output_path,
        use_blur_detection=use_blur_detection,
    )


def face_recognition(
    image,
    dbface,
    pixel_threshold=IMAGE_PIXEL_THRESHOLD,
    output=True,
    output_path=None,
    base_name=None,
    image_path=None,
    video_output_path=None,
    use_blur_detection=False,
):
    """
    人脸识别算法
    :param img:
    :param output
    :param output_path
    :param base_name:
    :return:
    """
    face_img_list = []
    face_path_list = []
    coordinate_list = []
    dimensions = image.shape
    logger.info("image shape={}".format(dimensions))
    if image is not None:
        # 判断图像通道数
        if len(dimensions) == 3 and dimensions[2] >= 3:
            image = image
            # logger.info("Image already has 3 or more channels.")
        elif len(dimensions) == 2:
            # 单通道灰度图像，转换为三通道BGR图像
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            logger.info(
                "Image was single-channel (grayscale), converted to 3 channels (BGR)."
            )
        elif len(dimensions) == 3 and dimensions[2] == 1:
            # 单通道图像（二维数组形式），转换为三通道BGR图像
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            logger.info(
                "Image was single-channel (grayscale), converted to 3 channels (BGR)."
            )
        elif len(dimensions) == 3 and dimensions[2] == 2:
            # 两通道图像，不是标准的图像格式，根据具体需求处理
            # 这里假设将其扩展为三通道（复制一个通道）
            image = cv2.merge([image, image[:, :, 0], image[:, :, 1]])
            logger.info(
                "Image was two-channel, converted to 3 channels by duplication."
            )
    else:
        error_msg = "输入图片错误，请检查图片路径是否正确。"
        logger.error(error_msg)
        raise BizException(error_msg)

    # 使用dbface检测人脸，并返回对应人脸的(x, y, w, h)值的list
    face_rects, img, ratio = detect_image(
        model=dbface, image=image, pixel_threshold=pixel_threshold
    )  # 返回的img为经过resize后的全图

    # 大于0则检测到人脸
    if len(face_rects) > 0:
        # 单独框出每一张人脸
        for index, face_rect in enumerate(face_rects):
            x, y, w, h = face_rect
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            face = img[int(y) : int(y + h), int(x) : int(x + w), :]
            # 对视频人脸，检测人脸是否模糊
            if pixel_threshold == VIDEO_PIXEL_THRESHOLD and use_blur_detection:
                if w * h <= VEDIO_EXPECTED_PIXELS:
                    fix_face = fix_image_size(
                        face, expected_pixels=VEDIO_EXPECTED_PIXELS
                    )
                    _, score, blurry = estimate_blur(fix_face, threshold=BLUR_THRESHOLD)
                else:
                    _, score, blurry = estimate_blur(face, threshold=BLUR_THRESHOLD)
                # 模糊人脸，跳过
                if blurry:
                    continue
            # 对图片人脸，不检测人脸是否模糊
            elif pixel_threshold == IMAGE_PIXEL_THRESHOLD and use_blur_detection:
                if w * h <= IMAGE_EXPECTED_PIXELS:
                    fix_face = fix_image_size(
                        face, expected_pixels=IMAGE_EXPECTED_PIXELS
                    )
                    _, _, blurry = estimate_blur(fix_face, threshold=1000)
                else:
                    _, _, blurry = estimate_blur(face, threshold=1000)
                # 模糊人脸，跳过
                if blurry:
                    continue
            # cv2格式转换成Image格式
            face_image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

            if output and base_name:
                splits = base_name.split(".")
                # index从 1 开始
                face_name = splits[0] + "_" + str(index + 1) + "." + splits[1]
                if not output_path:
                    output_path = video_output_path
                face_path = os.path.join(output_path, face_name)
                cv2.imwrite(face_path, face)
                face_path_list.append(face_path)
            face_img_list.append(face_image)
            coordinate_list.append(
                [
                    (int(x / ratio), int(y / ratio)),
                    (int((x + w) / ratio), int((y + h) / ratio)),
                ]
            )
    return face_img_list, face_path_list, coordinate_list


def fix_image_size(image: np.array, expected_pixels: float = 4e3):
    ratio = np.sqrt(expected_pixels / (image.shape[0] * image.shape[1]))
    return cv2.resize(image, (0, 0), fx=ratio, fy=ratio)


def estimate_blur(image: np.array, threshold: int = BLUR_THRESHOLD):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    score = np.var(blur_map)
    return blur_map, score, bool(score < threshold)


if __name__ == "__main__":
    image_path = ""

    face_image = face_recognition_path(image_path, dbface)
    n = 0
    for image in face_image:
        image.save(f"save_face_image/{n}.jpg")
        n += 1
    print(f"Program End...")
