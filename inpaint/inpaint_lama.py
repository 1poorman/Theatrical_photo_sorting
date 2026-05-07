import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


class ImageInpainter:
    def __init__(self, model_path: str):
        """
        初始化图像修复管道。
        
        :param model_path: 模型路径
        """
        if not model_path:
            model_path = '/home/huachenghao/codes/cv_fft_inpainting_lama'
        self.inpainting_pipeline = pipeline(Tasks.image_inpainting, model=model_path)

    def inpaint(self, img_path: str, mask_path: str) -> cv2.Mat:
        """
        执行图像修复。
        
        :param img_path: 原始图像路径
        :param mask_path: 掩码图像路径（黑白图，白色表示需修复区域）
        :return: 修复后的图像（OpenCV 格式，BGR）
        """
        input_data = {
            'img': img_path,
            'mask': mask_path,
        }
        result = self.inpainting_pipeline(input_data)
        vis_img = result[OutputKeys.OUTPUT_IMG]
        if vis_img is not None:
            print(f"Visualized result saved in {vis_img}")
        return vis_img


# 使用示例：
if __name__ == "__main__":
    model_path='/home/huachenghao/codes/cv_fft_inpainting_lama'
    inpainter = ImageInpainter(model_path)
    output_img = inpainter.inpaint(
        img_path='../test_images/1.JPG',
        mask_path='../output/detection/image-1/output_no_persons.jpg'
    )
    cv2.imwrite('result.png', output_img)