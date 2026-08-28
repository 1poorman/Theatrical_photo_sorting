import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image, ImageDraw, ImageFont

class VisualizationUtils:
    """
    可视化工具类
    包含人脸检测结果、特征向量、识别结果的可视化
    """
    
    @staticmethod
    def draw_bounding_boxes(image: np.ndarray, 
                           faces: List[Dict],
                           color: Tuple[int, int, int] = (0, 255, 0),
                           thickness: int = 2,
                           show_score: bool = True,
                           show_landmarks: bool = True) -> np.ndarray:
        """
        在图像上绘制人脸边界框
        
        Args:
            image: 输入图像
            faces: 人脸检测结果列表
            color: 边界框颜色
            thickness: 线宽
            show_score: 是否显示置信度
            show_landmarks: 是否显示关键点
        
        Returns:
            绘制后的图像
        """
        result = image.copy()
        
        for face in faces:
            bbox = face.get('bbox', [])
            score = face.get('score', 0.0)
            landmarks = face.get('landmarks', [])
            
            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                
                # 绘制边界框
                cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
                
                # 显示置信度
                if show_score:
                    label = f"{score:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    
                    # 绘制标签背景
                    cv2.rectangle(result, (x1, y1 - label_size[1] - 5),
                                 (x1 + label_size[0], y1), color, cv2.FILLED)
                    
                    # 绘制标签文本
                    cv2.putText(result, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # 绘制关键点
                if show_landmarks and landmarks:
                    for i, (x, y) in enumerate(landmarks):
                        cv2.circle(result, (int(x), int(y)), 2, (0, 0, 255), -1)
                        
                        # 可选：显示关键点编号
                        # cv2.putText(result, str(i), (int(x), int(y)),
                        #            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return result
    
    @staticmethod
    def draw_recognition_results(image: np.ndarray,
                                results: List[Dict],
                                threshold: float = 0.7,
                                show_distance: bool = True) -> np.ndarray:
        """
        绘制人脸识别结果
        
        Args:
            image: 输入图像
            results: 识别结果列表
            threshold: 置信度阈值
            show_distance: 是否显示距离
        
        Returns:
            绘制后的图像
        """
        result = image.copy()
        
        for i, face_result in enumerate(results):
            bbox = face_result.get('bbox', [])
            matches = face_result.get('matches', [])
            
            if len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                
                if matches and matches[0]['score'] >= threshold:
                    # 识别成功，绿色框
                    color = (0, 255, 0)
                    person_name = matches[0]['person_name']
                    score = matches[0]['score']
                    
                    label = f"{person_name}"
                    if show_distance:
                        label += f" ({score:.3f})"
                else:
                    # 识别失败或未知人脸，红色框
                    color = (0, 0, 255)
                    label = "Unknown"
                
                # 绘制边界框
                cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签背景
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness)
                
                cv2.rectangle(result, (x1, y1 - text_height - 10),
                             (x1 + text_width, y1), color, cv2.FILLED)
                
                # 绘制标签文本
                cv2.putText(result, label, (x1, y1 - 5),
                           font, font_scale, (255, 255, 255), thickness)
        
        return result
    
    @staticmethod
    def create_embedding_visualization(embeddings: np.ndarray,
                                      labels: List[str],
                                      title: str = "Face Embeddings Visualization",
                                      method: str = 'tsne') -> plt.Figure:
        """
        创建特征向量可视化
        
        Args:
            embeddings: 特征向量矩阵 (n_samples, n_features)
            labels: 标签列表
            title: 图表标题
            method: 降维方法 ['tsne', 'pca', 'umap']
        
        Returns:
            matplotlib Figure对象
        """
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        try:
            from umap import UMAP
        except ImportError:
            UMAP = None
        
        # 降维到2D
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        elif method == 'pca':
            reducer = PCA(n_components=2)
            embeddings_2d = reducer.fit_transform(embeddings)
        elif method == 'umap' and UMAP is not None:
            reducer = UMAP(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 获取唯一标签
        unique_labels = list(set(labels))
        colors = cm.get_cmap('tab20', len(unique_labels))
        
        # 为每个标签绘制点
        for i, label in enumerate(unique_labels):
            idx = [j for j, l in enumerate(labels) if l == label]
            points = embeddings_2d[idx]
            
            ax.scatter(points[:, 0], points[:, 1],
                      color=colors(i),
                      label=label,
                      alpha=0.7,
                      s=50)
        
        ax.set_title(title, fontsize=16)
        ax.set_xlabel(f"{method.upper()} Component 1", fontsize=12)
        ax.set_ylabel(f"{method.upper()} Component 2", fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_similarity_matrix(similarities: np.ndarray,
                              labels: List[str],
                              title: str = "Face Similarity Matrix") -> plt.Figure:
        """
        绘制相似度矩阵
        
        Args:
            similarities: 相似度矩阵 (n_samples, n_samples)
            labels: 标签列表
            title: 图表标题
        
        Returns:
            matplotlib Figure对象
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制热力图
        im = ax.imshow(similarities, cmap='YlOrRd', aspect='auto')
        
        # 设置坐标轴
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        
        # 添加颜色条
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Similarity', rotation=-90, va="bottom")
        
        # 添加标题
        ax.set_title(title, fontsize=16, pad=20)
        
        # 添加网格
        ax.set_xticks(np.arange(len(labels) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(labels) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_dashboard(image: np.ndarray,
                        face_results: List[Dict],
                        recognition_results: List[Dict],
                        aligned_faces: List[np.ndarray] = None) -> np.ndarray:
        """
        创建识别结果仪表板
        
        Args:
            image: 原始图像
            face_results: 人脸检测结果
            recognition_results: 识别结果
            aligned_faces: 对齐后的人脸图像
        
        Returns:
            仪表板图像
        """
        # 创建检测结果图像
        detection_img = VisualizationUtils.draw_bounding_boxes(
            image, face_results, show_score=True, show_landmarks=True)
        
        # 创建识别结果图像
        recognition_img = VisualizationUtils.draw_recognition_results(
            image, recognition_results, threshold=0.7)
        
        # 创建对齐人脸网格
        if aligned_faces:
            grid_img = VisualizationUtils.create_face_grid(aligned_faces)
        else:
            grid_img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # 调整图像大小
        detection_img = cv2.resize(detection_img, (640, 480))
        recognition_img = cv2.resize(recognition_img, (640, 480))
        grid_img = cv2.resize(grid_img, (640, 480))
        
        # 垂直拼接
        top_row = np.hstack([detection_img, recognition_img])
        dashboard = np.vstack([top_row, grid_img])
        
        return dashboard
    
    @staticmethod
    def create_face_grid(face_images: List[np.ndarray],
                        grid_size: Tuple[int, int] = None) -> np.ndarray:
        """
        创建人脸网格
        
        Args:
            face_images: 人脸图像列表
            grid_size: 网格大小 (rows, cols)
        
        Returns:
            网格图像
        """
        if not face_images:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        n_images = len(face_images)
        
        if grid_size is None:
            # 自动计算网格大小
            cols = int(np.ceil(np.sqrt(n_images)))
            rows = int(np.ceil(n_images / cols))
        else:
            rows, cols = grid_size
        
        # 获取最大图像尺寸
        max_h = max(img.shape[0] for img in face_images)
        max_w = max(img.shape[1] for img in face_images)
        
        # 创建网格
        grid_h = rows * max_h
        grid_w = cols * max_w
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        # 填充网格
        for i, face_img in enumerate(face_images):
            row = i // cols
            col = i % cols
            
            h, w = face_img.shape[:2]
            y_start = row * max_h + (max_h - h) // 2
            x_start = col * max_w + (max_w - w) // 2
            
            grid[y_start:y_start+h, x_start:x_start+w] = face_img
        
        return grid
    
    @staticmethod
    def add_text_with_background(image: np.ndarray,
                                text: str,
                                position: Tuple[int, int],
                                font_scale: float = 1.0,
                                thickness: int = 2,
                                text_color: Tuple[int, int, int] = (255, 255, 255),
                                bg_color: Tuple[int, int, int] = (0, 0, 0),
                                padding: int = 5) -> np.ndarray:
        """
        在图像上添加带背景的文字
        
        Args:
            image: 输入图像
            text: 要添加的文字
            position: 文字位置 (x, y)
            font_scale: 字体缩放
            thickness: 线宽
            text_color: 文字颜色
            bg_color: 背景颜色
            padding: 内边距
        
        Returns:
            添加文字后的图像
        """
        result = image.copy()
        x, y = position
        
        # 获取文字大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness)
        
        # 绘制背景矩形
        cv2.rectangle(result,
                     (x - padding, y - text_height - padding),
                     (x + text_width + padding, y + padding),
                     bg_color, cv2.FILLED)
        
        # 绘制文字
        cv2.putText(result, text, (x, y),
                   font, font_scale, text_color, thickness)
        
        return result
    
    @staticmethod
    def create_animated_gif(images: List[np.ndarray],
                           output_path: str,
                           duration: int = 500,
                           loop: int = 0) -> None:
        """
        创建GIF动画
        
        Args:
            images: 图像列表
            output_path: 输出路径
            duration: 每帧持续时间(毫秒)
            loop: 循环次数 (0表示无限循环)
        """
        pil_images = []
        for img in images:
            if len(img.shape) == 3:
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.fromarray(img)
            pil_images.append(pil_img)
        
        pil_images[0].save(output_path,
                          save_all=True,
                          append_images=pil_images[1:],
                          duration=duration,
                          loop=loop)
    
    @staticmethod
    def visualize_embedding_space(query_embedding: np.ndarray,
                                 database_embeddings: np.ndarray,
                                 database_labels: List[str],
                                 top_k: int = 10) -> plt.Figure:
        """
        可视化查询向量在嵌入空间中的位置
        
        Args:
            query_embedding: 查询向量
            database_embeddings: 数据库向量
            database_labels: 数据库标签
            top_k: 显示最相似的k个结果
        
        Returns:
            matplotlib Figure对象
        """
        from sklearn.manifold import TSNE
        
        # 合并所有向量
        all_embeddings = np.vstack([query_embedding.reshape(1, -1), database_embeddings])
        
        # 使用t-SNE降维
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(all_embeddings)
        
        # 分离查询向量和数据库向量
        query_2d = embeddings_2d[0]
        db_2d = embeddings_2d[1:]
        
        # 计算相似度
        similarities = np.dot(database_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 绘制所有数据库点
        scatter = ax.scatter(db_2d[:, 0], db_2d[:, 1],
                           c=similarities,
                           cmap='viridis',
                           alpha=0.6,
                           s=50)
        
        # 绘制查询点
        ax.scatter(query_2d[0], query_2d[1],
                  color='red',
                  marker='*',
                  s=300,
                  label='Query',
                  edgecolors='black')
        
        # 绘制top-k结果
        for idx in top_indices:
            ax.scatter(db_2d[idx, 0], db_2d[idx, 1],
                      color='orange',
                      s=100,
                      edgecolors='black',
                      linewidth=2)
            
            # 添加标签
            ax.annotate(database_labels[idx],
                       (db_2d[idx, 0], db_2d[idx, 1]),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=9)
        
        # 添加颜色条
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Similarity to Query', rotation=270, labelpad=15)
        
        ax.set_title('Face Embedding Space Visualization', fontsize=16)
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    @staticmethod
    def draw_faces_on_image(image, results):
        """
        在图像上绘制人脸框和标签
        Args:
            image: 原始图像
            results: 人脸识别结果列表
        Returns:
            annotated_image: 带有人脸框和标签的图像
        """
        # 复制原始图像以避免修改原图
        annotated_image = image.copy()
        
        # 为每个检测到的人脸绘制框和标签
        for i, result in enumerate(results):
            # 获取边界框坐标
            x1, y1, x2, y2 = result['bbox']
            
            # 确保坐标在图像范围内
            h, w = annotated_image.shape[:2]
            x1 = max(0, min(w-1, x1))
            y1 = max(0, min(h-1, y1))
            x2 = max(0, min(w-1, x2))
            y2 = max(0, min(h-1, y2))
            
            # 绘制边界框
            color = (0, 255, 0) if result['is_known'] else (0, 165, 255)  # 绿色表示已知，橙色表示未知
            thickness = 2
            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            
            # 准备标签文本
            if result['is_known']:
                # label = f"Face {i+1}: {result['identified_as']} ({result['identification_confidence']:.2f})"
                label = f"Face {i+1}"
            else:
                label = f"Face {i+1}: Unknown"
            
            # 增大字体大小
            font_scale = 1.5
            font_thickness = 4
            
            # 计算标签尺寸并绘制背景矩形
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            cv2.rectangle(annotated_image, 
                        (int(x1), int(y1) - text_height - 15), 
                        (int(x1) + text_width, int(y1)), 
                        color, -1)
            
            # 绘制标签文本
            cv2.putText(annotated_image, label, 
                    (int(x1), int(y1) - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

        # 将图像最长边限制在1024像素以内，保持宽高比
        height, width = annotated_image.shape[:2]
        max_dimension = max(height, width)
        
        if max_dimension > 1024:
            scale_factor = 1024 / max_dimension
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            annotated_image = cv2.resize(annotated_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        
        return annotated_image