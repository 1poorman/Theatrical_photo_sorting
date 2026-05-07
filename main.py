import os, cv2
import os.path as osp
import torch
import sys
sys.path.append(".")
import argparse

from detection.PersonMaskCreator import PersonMaskCreator
from inpaint.inpaint_lama import ImageInpainter
from seg_clothes.ClothesSegment import ClothesSegmenter
from embedding.image_search_system_module import ImageEmbedder, VectorDatabase, ImageSearcher, build_index, search_image, find_similar_image_groups_from_folder



"""
步骤：
0. 图像质量评估、选取
1. 调用PersonMaskCreator，检测并获得结果：results、mask
2. 人脸/服装检测，基于results边界框缩小范围后检测分割，获得不同人物的面目和服装
3. 识别人物（待定）
4. 原图-mask后修复，获得纯背景
5. 背景识别（待定）
6. 背景图片embedding，聚类
"""


device = "cuda:0" if torch.cuda.is_available() else "cpu"

def parse():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", default="config/PixelHacker_sdvae_f8d4.yaml")
    parser.add_argument("--image_folder", default="/home/huachenghao/codes/NCPA照片档案与视频【测试用】/歌剧《假面舞会》/【整理完成】歌剧《假面舞会》A组彩排")             # 图片文件夹，用于建立索引
    parser.add_argument("--image_cluster_folder", default="/home/huachenghao/codes/NCPA照片档案与视频【测试用】/歌剧《假面舞会》/【原始3】20170522歌剧院-歌剧《假面舞会》A组彩排-摄影王小京")     # 待聚类图片文件夹
    parser.add_argument("--image_dir", default="./test_images/2.jpg") # 待检测图片
    parser.add_argument("--query_image_dir", default="./test_images/2.jpg")       # 待查询图片位置
    parser.add_argument("--out_dir", default="./outputs")            # 输出文件夹
    parser.add_argument("--detection_model_path", default="/home/huachenghao/codes/Sorting_theatrical-photo/detection/yolo11l.pt")              # 检测模型权重位置
    parser.add_argument("--segclothes_model_path", default="/home/huachenghao/codes/clothes/models--mattmdjaga--segformer_b2_clothes/snapshots/584abc1e1d260e23c0fc627c5217a09b2b461046")       # 服装分割模型权重位置
    parser.add_argument("--inpainter_model_path", default="/home/huachenghao/codes/cv_fft_inpainting_lama")  # 修复模型权重位置
    parser.add_argument("--model_name", default="dinov2_small")
    
    return parser.parse_args()

if __name__ == "__main__":

    args = parse()

    # Create output directory if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)

    # # 检测
    # detection_model = PersonMaskCreator(args.detection_model_path)

    # results = detection_model.detect_persons_in_image(args.image_dir)

    # # Generate proper output file path for mask
    # image_filename = os.path.basename(args.image_dir)
    # mask_filename = f"mask_{os.path.splitext(image_filename)[0]}.png"
    # mask_output_path = os.path.join(args.out_dir, mask_filename)
    
    # mask = detection_model.generate_and_save_mask_from_results(args.image_dir, results, mask_output_path)  

    # # 服装分割
    # segclothes_model = ClothesSegmenter(args.segclothes_model_path)
    # pred_seg = segclothes_model.segment_with_yolo(args.image_dir, results)
    # extracted_img = segclothes_model.extract_segmented_area(args.image_dir, pred_seg)
    # #保存结果
    # # os.makedirs(output_dir, exist_ok=True)
    # extract_filename = f"extract_{os.path.splitext(image_filename)[0]}.png"
    # cv2.imwrite(os.path.join(args.out_dir, extract_filename), extracted_img)
    # # cv2.imwrite("extracted_img.png", extracted_img)
    # print("extracted_img saved successfully.")

    # # 图像修复
    # inpainter_model = ImageInpainter(args.inpainter_model_path)
    # inpainted_image = inpainter_model.inpaint(args.image_dir, mask_output_path)

    #  # Save inpainted image
    # inpaint_name = f"inpainted_{os.path.splitext(image_filename)[0]}.png" 
    # inpainted_output_path = os.path.join(args.out_dir, inpaint_name)
    # cv2.imwrite(inpainted_output_path, inpainted_image)
    # print("inpainted_image saved successfully.")


    # Build index
    # vector_db, embedder = build_index(args.image_folder, "./embedding/index", args.model_name)

    # # Search
    # results_search = search_image(args.query_image_dir, "./embedding/index")

    # merged_img = create_results_image(query_path, results, args.out_dir)
    # merged_img_name = f"merged_{os.path.splitext(image_filename)[0]}.png"
    # cv2.imwrite(os.path.join(args.out_dir, merged_img_name), merged_img)
    print("search_img saved successfully.")

    # Automatic clustering with default parameters
    groups = find_similar_image_groups_from_folder(args.image_cluster_folder, args.model_name)
    print("similar_img_groups saved successfully.")


    # Custom minimum cluster size
    # groups = find_similar_image_groups_from_folder("./images", model_name="blip2", min_cluster_size=5)






    

