import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ========== 1. 定义调色板（BGR 格式，共 18 类）==========
# 背景 (0) 用黑色 [0,0,0]，其他类使用鲜明颜色
PALETTE = [
    [0, 0, 0],        # 0: Background
    [128, 0, 128],    # 1: Hat (purple)
    [255, 255, 0],    # 2: Hair (cyan)
    [0, 255, 255],    # 3: Sunglasses (yellow)
    [0, 0, 255],      # 4: Upper-clothes (red)
    [255, 0, 255],    # 5: Skirt (magenta)
    [0, 255, 0],      # 6: Pants (lime)
    [255, 165, 0],    # 7: Dress (orange)
    [255, 255, 255],  # 8: Belt (white)
    [0, 128, 255],    # 9: Left-shoe (orange-blue)
    [255, 128, 0],    # 10: Right-shoe (blue-orange)
    [255, 255, 255],  # 11: Face (white) — 或可改为 skin tone
    [128, 255, 0],    # 12: Left-leg (green-yellow)
    [0, 255, 128],    # 13: Right-leg (green-cyan)
    [255, 0, 128],    # 14: Left-arm (pink)
    [128, 0, 255],    # 15: Right-arm (violet)
    [128, 128, 0],    # 16: Bag (olive)
    [0, 128, 128],    # 17: Scarf (teal)
]

# 确保 PALETTE 长度为 18
assert len(PALETTE) == 18, "Palette must have 18 colors"
model_dir = '/home/huachenghao/codes/clothes/models--mattmdjaga--segformer_b2_clothes/snapshots/584abc1e1d260e23c0fc627c5217a09b2b461046'
image_path = os.path.join(PROJECT_ROOT, 'data/sample_images/2.jpg')

processor = SegformerImageProcessor.from_pretrained(model_dir)
model = AutoModelForSemanticSegmentation.from_pretrained(model_dir)

# ========== 2. 读取图像和模型推理==========

image = cv2.imread(image_path)
height, width = image.shape[:2]

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits.cpu()

upsampled_logits = F.interpolate(
    logits,
    size=(height, width),
    mode="bilinear",
    align_corners=False,
)

# 获取预测类别图 [H, W]
pred_seg = upsampled_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

# ========== 3. 生成彩色分割图 ==========
color_seg = np.zeros((height, width, 3), dtype=np.uint8)
for class_id in range(len(PALETTE)):
    color_seg[pred_seg == class_id] = PALETTE[class_id]  # BGR

# ========== 4. 叠加：背景用原图，前景用 color_seg ==========
# 创建 mask：背景区域为 False，前景为 True
mask = (pred_seg > 0)[..., None]  # shape (H, W, 1) 用于广播

# 混合：原图 * (1 - mask) + color_seg * mask
overlay = image * (1 - mask) + color_seg * mask
overlay = overlay.astype(np.uint8)

# ========== 5. 保存结果 ==========
cv2.imwrite(os.path.join(PROJECT_ROOT, "outputs/seg_clothes/image-2/output_overlay.jpg"), overlay)

# （可选）单独保存纯分割图
cv2.imwrite(os.path.join(PROJECT_ROOT, "outputs/seg_clothes/image-2/seg_color.png"), color_seg)

# ========== 6. 保存选定类别分割结果 ==========
# 只对 Hat (1), Upper-clothes (4), Pants (6) 上色
interested_classes = [1, 4, 6]


# 如果你想让未选中的类别保持原图，可以这样操作：
mask_interested = np.isin(pred_seg, interested_classes)[..., None]
overlay_selected = image * (~mask_interested) + color_seg * mask_interested
overlay_selected = overlay_selected.astype(np.uint8)

cv2.imwrite(os.path.join(PROJECT_ROOT, "outputs/seg_clothes/image-2/output_overlay_selected_classes.jpg"), overlay_selected)
if overlay_selected is None:
    print("output_overlay_selected_classes.jpg is saved.")