# -*- coding: utf-8 -*-
"""role_classifier - 戏曲行当 zero-shot 分类（F5）

基于 SigLIP2 图文对齐：对主角人物区域（人物框裁剪）与生/旦/净/丑多模板文本
计算余弦相似度，softmax 得行当概率。置信度 < 0.4 标记 uncertain。

不训练新模型；低置信结果仅入报告，不写入整理文件名。
"""
import cv2
import numpy as np
import torch

from core_modules.tools.logger import get_app_logger

logger = get_app_logger()

# 中英双语多模板（同角色组内平均；模板间强调互斥视觉特征）
ROLE_TEMPLATES = {
    '生': [
        'a middle-aged or young male Chinese opera actor with a natural face, '
        'light makeup, no face paint, wearing a simple black beard',
        'a Peking opera sheng role, natural skin face with normal eyebrows',
        '京剧生行演员，素脸无脸谱，男性角色',
    ],
    '旦': [
        'a young beautiful Chinese opera actress (dan role), delicate feminine makeup '
        'with rouge, and an elaborate headdress with pearls and flowers',
        'a Peking opera female performer with graceful look and long black hair pieces '
        'around the face',
        '京剧旦行演员，女性角色，贴片子头面，妆容娇艳',
    ],
    '净': [
        'a Chinese opera jing role, the entire face is painted with bold colorful '
        'lianpu patterns in black, red, white or blue covering forehead and cheeks',
        'a Peking opera painted face warrior with fully painted dramatic facial patterns',
        '京剧净行花脸演员，整脸勾画黑红白色脸谱',
    ],
    '丑': [
        'a Chinese opera chou clown role with a distinctive small white square patch '
        'painted around the nose and eyes on an otherwise natural face',
        'a comedic Peking opera chou actor with white butterfly-shaped patch on the nose',
        '京剧丑行演员，鼻梁眼窝勾画白色豆腐块',
    ],
}

UNCERTAIN_THRESHOLD = 0.4


class RoleClassifier:
    def __init__(self, image_embedder):
        """
        Args:
            image_embedder: ImageEmbedder（siglip2 系，用于图像特征）
        """
        self.embedder = image_embedder
        if not image_embedder.model_name.startswith('siglip'):
            raise ValueError('RoleClassifier 需要 siglip2 系 ImageEmbedder')
        self.device = image_embedder.device
        self.text_feats = self._encode_templates()

    def _encode_templates(self):
        """编码行当模板文本（组内平均后归一化）。"""
        from transformers import AutoTokenizer
        processor = self.embedder.processor
        tokenizer = AutoTokenizer.from_pretrained(
            self.embedder.processor.tokenizer.name_or_path)

        role_feats = {}
        for role, templates in ROLE_TEMPLATES.items():
            # SigLIP 训练时文本为 max_length padding（EOS 位置敏感），必须保持一致
            inputs = tokenizer(templates, padding='max_length', max_length=64,
                               truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                # 兼容 transformers 5.x（返回输出对象）与旧版（直接返回张量）
                out = self.embedder.model.get_text_features(**inputs)
                if hasattr(out, 'pooler_output'):
                    out = out.pooler_output
            out = torch.nn.functional.normalize(out, p=2, dim=-1)
            role_feats[role] = out.mean(dim=0)
        # 归一化角色均值向量
        roles = list(role_feats)
        mat = torch.stack([role_feats[r] for r in roles])
        mat = torch.nn.functional.normalize(mat, p=2, dim=-1)
        return roles, mat.cpu().numpy()

    def classify_crop(self, crop_bgr):
        """对人物裁剪区域分类。

        Returns:
            dict: {role, confidence, probabilities}
        """
        from PIL import Image
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        # 复用 SigLIP 单图前向（接收 PIL 图像）
        img_feat = self.embedder._extract_features_siglip(img)
        if img_feat is None:
            return None
        img_feat = np.asarray(img_feat).flatten()
        sims = self.text_feats[1] @ img_feat
        # softmax 得概率（温度 100，同 CLIP zero-shot 惯例）
        probs = np.exp(100 * sims) / np.exp(100 * sims).sum()
        i = int(np.argmax(probs))
        role, conf = self.text_feats[0][i], float(probs[i])
        return {
            'role': role if conf >= UNCERTAIN_THRESHOLD else 'uncertain',
            'confidence': round(conf, 4),
            'probabilities': {r: round(float(p), 4) for r, p in zip(self.text_feats[0], probs)},
        }

    def classify_image(self, img, main_bbox=None, face_bbox=None):
        """对整图分类：优先用主角人物框，其次最大人脸框外扩。"""
        h, w = img.shape[:2]
        box = main_bbox
        if box is None and face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            fw, fh = x2 - x1, y2 - y1
            box = [max(0, int(x1 - fw)), max(0, int(y1 - fh * 0.6)),
                   min(w, int(x2 + fw)), min(h, int(y2 + fh * 1.5))]
        if box is None:
            return None
        x1, y1, x2, y2 = [int(v) for v in box]
        crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if crop.size == 0:
            return None
        return self.classify_crop(crop)
