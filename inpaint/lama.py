import cv2
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

input_location = '/home/huachenghao/codes/NCPA照片档案与视频【测试用】/歌剧《假面舞会》/【整理完成】歌剧《假面舞会》A组彩排/10-20170522-歌剧院-歌剧《假面舞会》A组彩排 第1幕1场-左起：朱塞佩·季帕里饰古斯塔夫三世、谢尔盖·穆尔扎耶夫饰雷纳托-王小京.JPG'
input_mask_location = '/home/huachenghao/codes/ultralytics-main/tests/output_no_persons.jpg'
input = {
        'img':input_location,
        'mask':input_mask_location,
}

inpainting = pipeline(Tasks.image_inpainting, model='/home/huachenghao/codes/cv_fft_inpainting_lama')
result = inpainting(input)
vis_img = result[OutputKeys.OUTPUT_IMG]
cv2.imwrite('result.png', vis_img)

