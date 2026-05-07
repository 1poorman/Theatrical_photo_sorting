# image_search_system.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import faiss
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

class ImageEmbedder:
    def __init__(self, model_name='resnet50', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_name = model_name
        self.model, self.feature_dim, self.processor = self._load_model(model_name)
        self.preprocess = self._get_preprocess(model_name)
        
    def _load_model(self, model_name):
        """加载预训练的模型"""
        if model_name.startswith('resnet'):
            return self._load_resnet(model_name)
        elif model_name.startswith('dinov2'):
            return self._load_dinov2(model_name)
        elif model_name.startswith('nomic'):
            return self._load_nomic(model_name)
        elif model_name.startswith('blip2'):
            return self._load_blip2(model_name)
        else:
            print(f"未知模型: {model_name}，使用默认的ResNet50")
            return self._load_resnet('resnet50')
    
    def _load_resnet(self, model_name):
        """加载ResNet模型"""
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            feature_dim = 2048
        elif model_name == 'resnet101':
            model = models.resnet101(pretrained=True)
            feature_dim = 2048
        else:
            model = models.resnet18(pretrained=True)
            feature_dim = 512
        
        # 移除最后的分类层，使用全局平均池化层的输出作为特征
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()
        model.to(self.device)
        return model, feature_dim, None
    
    def _load_dinov2(self, model_name):
        """加载DINOv2模型"""
        try:
            # 尝试从torch hub加载
            if model_name == 'dinov2_small':
                model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
                feature_dim = 384
            elif model_name == 'dinov2_base':
                model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
                feature_dim = 768
            elif model_name == 'dinov2_large':
                model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
                feature_dim = 1024
            else:  # dinov2_giant
                model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
                feature_dim = 1536
        except Exception as e:
            print(f"加载DINOv2模型失败: {e}")
            print("使用ResNet50作为替代")
            return self._load_resnet('resnet50')
        
        model.eval()
        model.to(self.device)
        return model, feature_dim, None
    
    def _load_nomic(self, model_name):
        """加载Nomic模型（本地版本）"""
        try:
            from transformers import AutoImageProcessor, AutoModel
            
            print("加载本地Nomic Embedding Vision模型...")
            
            # 加载处理器和模型
            processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
            model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)

            feature_dim = 768  # nomic-embed-vision-v1.5的特征维度
            
            model.eval()
            model.to(self.device)
            
            print("Nomic模型加载成功！")
            return model, feature_dim, processor
            
        except Exception as e:
            print(f"加载Nomic模型失败: {e}")
            print("使用ResNet50作为替代")
            # return self._load_resnet('resnet50')

    def _load_blip2(self, model_name):
        """加载BLIP2模型"""
        try:
            print("Loading BLIP2 model...")
            
            # Load BLIP2 processor and model
            processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
            
            # For BLIP2, we'll use the image embedding dimension
            feature_dim = 768  # CLIP vision encoder dimension
            
            model.eval()
            model.to(self.device)
            
            print("BLIP2 model loaded successfully!")
            return model, feature_dim, processor
        except Exception as e:
            print(f"Failed to load BLIP2 model: {e}")
            print("Falling back to ResNet50")
            return self._load_resnet('resnet50')
    
    def _get_preprocess(self, model_name):
        """定义图片预处理流程"""
        if model_name.startswith('dinov2'):
            # DINOv2的预处理
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        elif model_name.startswith('nomic'):
            # Nomic模型使用自己的处理器，这里返回一个占位符
            # 实际预处理在extract_features中通过processor处理
            return lambda x: x
        elif model_name.startswith('blip2'):
            # BLIP2使用自己的处理器，返回占位符
            return lambda x: x
        else:
            # ResNet的预处理
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def extract_features(self, image_path):
        """提取单张图片的特征向量"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.model_name.startswith('nomic'):
                # 使用Nomic本地模型提取特征
                return self._extract_features_nomic_local(image)
            elif self.model_name.startswith('blip2'):
                # 使用BLIP2模型提取特征
                return self._extract_features_blip2(image)
            else:
                # 使用本地模型提取特征
                input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    if self.model_name.startswith('dinov2'):
                        # DINOv2模型的特征提取
                        features = self.model(input_tensor)
                    else:
                        # ResNet模型的特征提取
                        features = self.model(input_tensor)
                        features = features.squeeze(-1).squeeze(-1)
                
                return features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def _extract_features_nomic_local(self, image):
        """使用本地Nomic模型提取特征"""
        try:
            # 使用Nomic处理器处理图像
            inputs = self.processor(image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # 前向传播
                img_emb = self.model(**inputs).last_hidden_state
                # 取第一个token的嵌入并归一化
                img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)
            
            return img_embeddings.cpu().numpy().flatten()
        except Exception as e:
            print(f"Nomic本地模型特征提取失败: {e}")
            return None

    def _extract_features_blip2(self, image):
        """使用BLIP2模型提取特征"""
        try:
            # Process image with BLIP2 processor
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Extract image embeddings
                outputs = self.model.vision_model(**inputs)
                # Use the pooled output (CLS token) as the image embedding
                image_embeds = outputs.pooler_output
                
                # Normalize the embeddings
                image_embeds = F.normalize(image_embeds, p=2, dim=-1)
            
            return image_embeds.cpu().numpy().flatten()
        except Exception as e:
            print(f"BLIP2 feature extraction failed: {e}")
            return None
    
    def extract_features_batch(self, image_paths):
        """批量提取特征，提高效率"""
        # 如果是Nomic模型，使用专门的批量处理
        if self.model_name.startswith('nomic'):
            images = []
            valid_paths = []
            
            # 加载所有图片
            for path in image_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    images.append(image)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue
            
            if not images:
                return [], []
            
            # 批量处理
            features = []
            for image in images:
                feat = self._extract_features_nomic_local(image)
                if feat is not None:
                    features.append(feat)
            
            return np.array(features), valid_paths

         # 如果是BLIP2模型，需要单独处理（不支持真正的批处理）
        elif self.model_name.startswith('blip2'):
            images = []
            valid_paths = []
            
            # 加载所有图片
            for path in image_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    images.append(image)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue
            
            if not images:
                return [], []
            
            # 处理每个图像
            features = []
            for image in images:
                feat = self._extract_features_blip2(image)
                if feat is not None:
                    features.append(feat)
            
            return np.array(features), valid_paths
        
        # 其他模型的批量处理
        images = []
        valid_paths = []
        
        # 预处理所有图片
        for path in image_paths:
            try:
                image = Image.open(path).convert('RGB')
                input_tensor = self.preprocess(image)
                images.append(input_tensor)
                valid_paths.append(path)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                continue
        
        if not images:
            return [], []
        
        # 批量处理
        batch_tensor = torch.stack(images).to(self.device)
        
        with torch.no_grad():
            if self.model_name.startswith('dinov2'):
                # DINOv2模型的特征提取
                features = self.model(batch_tensor)
            else:
                # ResNet模型的特征提取
                features = self.model(batch_tensor)
                features = features.squeeze(-1).squeeze(-1)
        
        return features.cpu().numpy(), valid_paths


class VectorDatabase:
    def __init__(self, dimension=2048, model_name='resnet50'):
        self.dimension = dimension
        self.model_name = model_name
        self.index = None
        self.image_paths = []
        self.embeddings = []
        
    def build_index(self, image_folder, embedder, batch_size=32):
        """批量处理图片并建立索引"""
        print("开始提取图片特征...")
        
        # 获取所有图片路径
        image_files = self._get_image_files(image_folder)
        print(f"找到 {len(image_files)} 张图片")
        
        # 批量提取特征
        embeddings = []
        valid_paths = []
        
        for i in tqdm(range(0, len(image_files), batch_size)):
            batch_files = image_files[i:i+batch_size]
            
            for image_path in batch_files:
                features = embedder.extract_features(image_path)
                if features is not None:
                    embeddings.append(features)
                    valid_paths.append(image_path)
        
        self.embeddings = np.array(embeddings).astype('float32')
        self.image_paths = valid_paths
        
        print(f"成功提取 {len(self.embeddings)} 张图片的特征")
        
        # 建立FAISS索引
        self._build_faiss_index()
    
    def build_index_optimized(self, image_folder, embedder, batch_size=64):
        """优化版的索引构建，使用批量处理"""
        print("开始批量提取图片特征...")
        
        # 获取所有图片路径
        image_files = self._get_image_files(image_folder)
        print(f"找到 {len(image_files)} 张图片")
        
        # 批量处理
        all_embeddings = []
        all_paths = []
        
        for i in tqdm(range(0, len(image_files), batch_size)):
            batch_files = image_files[i:i+batch_size]
            features, valid_paths = embedder.extract_features_batch(batch_files)
            
            if len(features) > 0:
                all_embeddings.append(features)
                all_paths.extend(valid_paths)
        
        if all_embeddings:
            self.embeddings = np.vstack(all_embeddings).astype('float32')
            self.image_paths = all_paths
            self._build_faiss_index()
        else:
            print("没有成功提取任何图片特征")
    
    def _get_image_files(self, image_folder):
        """获取文件夹中的所有图片文件"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        for root, dirs, files in os.walk(image_folder):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        return image_files
        
    def _build_faiss_index(self):
        """使用FAISS建立向量索引"""
        print("建立FAISS索引...")
        
        # 使用内积相似度（余弦相似度，因为向量已经归一化）
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # 归一化向量以便使用内积计算余弦相似度
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        print(f"索引建立完成，包含 {self.index.ntotal} 个向量")
    
    def save_index(self, save_path):
        """保存索引和元数据"""
        os.makedirs(save_path, exist_ok=True)
        
        # 保存FAISS索引
        faiss.write_index(self.index, os.path.join(save_path, "vector_index.faiss"))
        
        # 保存图片路径
        with open(os.path.join(save_path, "image_paths.txt"), 'w', encoding='utf-8') as f:
            for path in self.image_paths:
                f.write(path + '\n')
        
        # 保存嵌入向量（可选）
        np.save(os.path.join(save_path, "embeddings.npy"), self.embeddings)
        
        # 保存模型信息
        with open(os.path.join(save_path, "model_info.txt"), 'w') as f:
            f.write(f"model: {self.model_name}\n")
            f.write(f"dimension: {self.dimension}\n")
        
        print(f"索引已保存到 {save_path}")
    
    def load_index(self, load_path):
        """加载索引和元数据"""
        # 加载FAISS索引
        self.index = faiss.read_index(os.path.join(load_path, "vector_index.faiss"))
        
        # 加载图片路径
        with open(os.path.join(load_path, "image_paths.txt"), 'r', encoding='utf-8') as f:
            self.image_paths = [line.strip() for line in f.readlines()]
        
        # 加载嵌入向量（可选）
        embeddings_path = os.path.join(load_path, "embeddings.npy")
        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)
        
        # 加载模型信息
        model_info_path = os.path.join(load_path, "model_info.txt")
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                for line in f:
                    if line.startswith('model:'):
                        self.model_name = line.split(':')[1].strip()
                    if line.startswith('dimension:'):
                        self.dimension = int(line.split(':')[1].strip())
        
        print(f"索引加载完成，包含 {self.index.ntotal} 个向量")


class ImageSearcher:
    def __init__(self, vector_db, embedder):
        self.vector_db = vector_db
        self.embedder = embedder
    
    def search(self, query_image_path, top_k=5):
        """搜索相似图片"""
        # 提取查询图片特征
        query_features = self.embedder.extract_features(query_image_path)
        if query_features is None:
            return []
        
        query_features = query_features.astype('float32').reshape(1, -1)
        
        # 归一化查询向量
        faiss.normalize_L2(query_features)
        
        # 搜索相似向量
        similarities, indices = self.vector_db.index.search(query_features, top_k)
        
        # 组织结果
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.vector_db.image_paths):  # 确保索引有效
                results.append({
                    'rank': i + 1,
                    'image_path': self.vector_db.image_paths[idx],
                    'similarity': float(similarity)
                })
        
        return results
    
    def search_by_image(self, query_image, top_k=5):
        """直接使用PIL Image对象进行搜索"""
        # 临时保存图片并搜索
        temp_path = "temp_query_image.jpg"
        query_image.save(temp_path)
        results = self.search(temp_path, top_k)
        
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return results


def create_results_image(query_path, results, save_path=None, max_width=1000):
    """创建查询图片和搜索结果的合并图片（优化布局）"""
    # 加载查询图片
    query_img = Image.open(query_path)
    
    # 调整查询图片大小 - 占据左侧1/3宽度
    query_width, query_height = query_img.size
    query_display_width = max_width // 3
    query_display_height = int(query_display_width * query_height / query_width)
    query_img = query_img.resize((query_display_width, query_display_height), Image.Resampling.LANCZOS)
    
    # 加载结果图片并调整大小
    result_imgs = []
    for result in results[:5]:  # 只取前5个结果
        img = Image.open(result['image_path'])
        img_width, img_height = img.size
        
        # 每个结果图片占据右侧2/3宽度的1/4（即每行3个图片）
        result_display_width = (max_width - query_display_width) // 4
        result_display_height = int(result_display_width * img_height / img_width)
        img = img.resize((result_display_width, result_display_height), Image.Resampling.LANCZOS)
        result_imgs.append((img, result['similarity'], result['rank']))
    
    # 计算合并图片的高度
    # 查询图片高度 + 两行结果图片高度 + 间距和标题
    row_height = max([img.height for img, _, _ in result_imgs]) if result_imgs else 0
    total_height = max(query_display_height, row_height * 2) + 150  # 添加标题和间距空间
    
    # 创建合并图片
    merged_img = Image.new('RGB', (max_width, total_height), 'white')
    
    # 绘制查询图片（左侧居中）
    query_y = (total_height - query_display_height) // 2
    merged_img.paste(query_img, (20, query_y))
    
    # 绘制结果图片（右侧分为两行）
    results_start_x = query_display_width + 40
    
    # 第一行：前3个结果
    for i in range(3):
        if i < len(result_imgs):
            img, similarity, rank = result_imgs[i]
            x = results_start_x + i * (result_imgs[0][0].width + 20)
            y = 40
            merged_img.paste(img, (x, y))
    
    # 第二行：后2个结果（居中显示）
    for i in range(3, 5):
        if i < len(result_imgs):
            img, similarity, rank = result_imgs[i]
            # 居中放置后两个图片
            row_width = 2 * result_imgs[0][0].width + 20
            start_x = results_start_x + (max_width - results_start_x - row_width) // 2
            x = start_x + (i - 3) * (result_imgs[0][0].width + 20)
            y = 40 + row_height + 40
            merged_img.paste(img, (x, y))
    
    # 添加文字说明
    draw = ImageDraw.Draw(merged_img)
    
    # 尝试加载字体，如果失败则使用默认字体
    try:
        font_large = ImageFont.truetype("arial.ttf", 24)
        font_medium = ImageFont.truetype("arial.ttf", 20)
        font_small = ImageFont.truetype("arial.ttf", 16)
    except:
        # 使用默认字体
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # 添加查询图片标题
    query_title = "查询图片"
    draw.text((query_display_width // 2 + 20, 30), query_title, fill='black', font=font_large, anchor='mm')
    
    # 添加结果图片标题
    results_title = "搜索结果"
    draw.text((results_start_x + (max_width - results_start_x) // 2, 30), results_title, fill='black', font=font_large, anchor='mm')
    
    # 添加结果图片的排名和相似度
    # 第一行
    for i in range(3):
        if i < len(result_imgs):
            _, similarity, rank = result_imgs[i]
            x = results_start_x + i * (result_imgs[0][0].width + 20) + result_imgs[0][0].width // 2
            y = 40 + row_height + 10
            
            rank_text = f"第{rank}名"
            similarity_text = f"相似度: {similarity:.3f}"
            
            draw.text((x, y), rank_text, fill='black', font=font_medium, anchor='mm')
            draw.text((x, y + 10 ), similarity_text, fill='black', font=font_small, anchor='mm')
    
    # 第二行
    for i in range(3, 5):
        if i < len(result_imgs):
            _, similarity, rank = result_imgs[i]
            # 居中放置后两个图片
            row_width = 2 * result_imgs[0][0].width + 20
            start_x = results_start_x + (max_width - results_start_x - row_width) // 2
            x = start_x + (i - 3) * (result_imgs[0][0].width + 20) + result_imgs[0][0].width // 2
            y = 40 + row_height + 40 + row_height + 20
            
            rank_text = f"第{rank}名"
            similarity_text = f"相似度: {similarity:.3f}"
            
            draw.text((x, y), rank_text, fill='black', font=font_medium, anchor='mm')
            draw.text((x, y + 15), similarity_text, fill='black', font=font_small, anchor='mm')
    
    # 保存或显示图片
    if save_path:
        merged_img.save(save_path)
        print(f"结果图片已保存到: {save_path}")
    
    return merged_img


def display_results(query_path, results, figsize=(15, 10), save_path=None):
    """可视化搜索结果"""
    # 创建合并图片
    merged_img = create_results_image(query_path, results, save_path)
    
    # 显示图片
    plt.figure(figsize=figsize)
    plt.imshow(merged_img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def create_web_searcher(index_path):
    """创建用于Web服务的搜索器"""
    # 从索引中获取模型信息
    model_info_path = os.path.join(index_path, "model_info.txt")
    model_name = "resnet50"  # 默认模型
    
    if os.path.exists(model_info_path):
        with open(model_info_path, 'r') as f:
            for line in f:
                if line.startswith('model:'):
                    model_name = line.split(':')[1].strip()
    
    embedder = ImageEmbedder(model_name=model_name)
    vector_db = VectorDatabase()
    vector_db.load_index(index_path)
    return ImageSearcher(vector_db, embedder)