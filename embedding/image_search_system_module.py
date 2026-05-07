# image_search_system_module.py
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
import datetime
from typing import List, Dict, Optional, Tuple, Union
from sklearn.cluster import KMeans

# Conditional imports for optional models
try:
    from transformers import Blip2Processor, Blip2Model
    BLIP2_AVAILABLE = True
except ImportError:
    BLIP2_AVAILABLE = False
    print("BLIP2 not available. Install transformers package to enable BLIP2 support.")

try:
    from transformers import AutoImageProcessor, AutoModel
    NOMIC_AVAILABLE = True
except ImportError:
    NOMIC_AVAILABLE = False
    print("Nomic model components not available.")



class ImageEmbedder:
    """
    Image feature extractor supporting multiple models including ResNet, DINOv2, Nomic, and BLIP2.
    """
    def __init__(self, model_name='resnet50', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_name = model_name
        self.model, self.feature_dim, self.processor = self._load_model(model_name)
        self.preprocess = self._get_preprocess(model_name)
        
    def _load_model(self, model_name):
        """Load pre-trained model based on model name"""
        if model_name.startswith('resnet'):
            return self._load_resnet(model_name)
        elif model_name.startswith('dinov2'):
            return self._load_dinov2(model_name)
        elif model_name.startswith('nomic'):
            return self._load_nomic(model_name) if NOMIC_AVAILABLE else self._load_resnet('resnet50')
        elif model_name.startswith('blip2'):
            return self._load_blip2(model_name) if BLIP2_AVAILABLE else self._load_resnet('resnet50')
        else:
            print(f"Unknown model: {model_name}, using default ResNet50")
            return self._load_resnet('resnet50')
    
    def _load_resnet(self, model_name):
        """Load ResNet model"""
        if model_name == 'resnet50':
            model = models.resnet50(weights=True)
            feature_dim = 2048
        elif model_name == 'resnet101':
            model = models.resnet101(weights=True)
            feature_dim = 2048
        else:
            model = models.resnet18(weights=True)
            feature_dim = 512
        
        # Remove classification layer and use global average pooling output as features
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()
        model.to(self.device)
        return model, feature_dim, None
    
    def _load_dinov2(self, model_name):
        """Load DINOv2 model with improved error handling"""
        try:
            print(f"Loading DINOv2 model: {model_name}")
            
            # Try to load from torch hub
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
                
            model.eval()
            model.to(self.device)
            return model, feature_dim, None
            
        except Exception as e:
            print(f"Failed to load DINOv2 model: {str(e)}")
            # print("Using ResNet50 as fallback")
            # return self._load_resnet('resnet50')
    
    def _load_nomic(self, model_name):
        """Load Nomic model (local version)"""
        try:
            print("Loading local Nomic Embedding Vision model...")
            
            # Load processor and model
            processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
            model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)

            feature_dim = 768  # nomic-embed-vision-v1.5 feature dimension
            
            model.eval()
            model.to(self.device)
            
            print("Nomic model loaded successfully!")
            return model, feature_dim, processor
            
        except Exception as e:
            print(f"Failed to load Nomic model: {e}")
            print("Using ResNet50 as fallback")
            return self._load_resnet('resnet50')
    
    def _load_blip2(self, model_name):
        """Load BLIP2 model"""
        if not BLIP2_AVAILABLE:
            print("BLIP2 not available. Please install transformers package.")
            return self._load_resnet('resnet50')
            
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
        """Define image preprocessing pipeline"""
        if model_name.startswith('dinov2'):
            # DINOv2 preprocessing
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        elif model_name.startswith('nomic') or model_name.startswith('blip2'):
            # Nomic/BLIP2 models use their own processors, return placeholder
            return lambda x: x
        else:
            # ResNet preprocessing
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def extract_features(self, image_path: str) -> Optional[np.ndarray]:
        """Extract features from a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            if self.model_name.startswith('nomic'):
                # Use local Nomic model for feature extraction
                return self._extract_features_nomic_local(image)
            elif self.model_name.startswith('blip2'):
                # Use BLIP2 model for feature extraction
                return self._extract_features_blip2(image)
            else:
                # Use local model for feature extraction
                input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    if self.model_name.startswith('dinov2'):
                        # DINOv2 model feature extraction
                        features = self.model(input_tensor)
                    else:
                        # ResNet model feature extraction
                        features = self.model(input_tensor)
                        features = features.squeeze(-1).squeeze(-1)
                
                return features.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None
    
    def _extract_features_nomic_local(self, image: Image.Image) -> Optional[np.ndarray]:
        """Extract features using local Nomic model"""
        try:
            # Process image with Nomic processor
            inputs = self.processor(image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Forward pass
                img_emb = self.model(**inputs).last_hidden_state
                # Take first token embedding and normalize
                img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)
            
            return img_embeddings.cpu().numpy().flatten()
        except Exception as e:
            print(f"Nomic local model feature extraction failed: {e}")
            return None

    def _extract_features_blip2(self, image: Image.Image) -> Optional[np.ndarray]:
        """Extract features using BLIP2 model"""
        if not BLIP2_AVAILABLE:
            return None
            
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
    
    def extract_features_batch(self, image_paths: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Batch feature extraction for efficiency"""
        # For Nomic model, use dedicated batch processing
        if self.model_name.startswith('nomic'):
            images = []
            valid_paths = []
            
            # Load all images
            for path in image_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    images.append(image)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue
            
            if not images:
                return np.array([]), []
            
            # Batch process
            features = []
            for image in images:
                feat = self._extract_features_nomic_local(image)
                if feat is not None:
                    features.append(feat)
            
            return np.array(features), valid_paths

        # For BLIP2 model, process individually (no true batching)
        elif self.model_name.startswith('blip2'):
            images = []
            valid_paths = []
            
            # Load all images
            for path in image_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    images.append(image)
                    valid_paths.append(path)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    continue
            
            if not images:
                return np.array([]), []
            
            # Process each image
            features = []
            for image in images:
                feat = self._extract_features_blip2(image)
                if feat is not None:
                    features.append(feat)
            
            return np.array(features), valid_paths
        
        # Batch processing for other models
        images = []
        valid_paths = []
        
        # Preprocess all images
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
            return np.array([]), []
        
        # Batch process
        batch_tensor = torch.stack(images).to(self.device)
        
        with torch.no_grad():
            if self.model_name.startswith('dinov2'):
                # DINOv2 model feature extraction
                features = self.model(batch_tensor)
            else:
                # ResNet model feature extraction
                features = self.model(batch_tensor)
                features = features.squeeze(-1).squeeze(-1)
        
        return features.cpu().numpy(), valid_paths


class VectorDatabase:
    """
    Vector database for storing and searching image embeddings using FAISS.
    """
    def __init__(self, dimension: int = 2048, model_name: str = 'resnet50'):
        self.dimension = dimension
        self.model_name = model_name
        self.index = None
        self.image_paths: List[str] = []
        self.embeddings: np.ndarray = np.array([])
        
    def build_index(self, image_folder: str, embedder: ImageEmbedder, batch_size: int = 32):
        """Process images in batches and build index"""
        print("Starting to extract image features...")
        
        # Get all image paths
        image_files = self._get_image_files(image_folder)
        print(f"Found {len(image_files)} images")
        
        # Batch feature extraction
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
        
        print(f"Successfully extracted features from {len(self.embeddings)} images")
        
        # Build FAISS index
        self._build_faiss_index()
    
    def build_index_optimized(self, image_folder: str, embedder: ImageEmbedder, batch_size: int = 64):
        """Optimized index building using batch processing"""
        print("Starting batch image feature extraction...")
        
        # Get all image paths
        image_files = self._get_image_files(image_folder)
        print(f"Found {len(image_files)} images")
        
        # Batch processing
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
            print("No image features were successfully extracted")
    
    def _get_image_files(self, image_folder: str) -> List[str]:
        """Get all image files in folder"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        image_files = []
        for root, dirs, files in os.walk(image_folder):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))
        return image_files
        
    def _build_faiss_index(self):
        """Build FAISS vector index"""
        print("Building FAISS index...")
        
        # Use inner product similarity (cosine similarity since vectors are normalized)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Normalize vectors for cosine similarity calculation using inner product
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        print(f"Index built successfully with {self.index.ntotal} vectors")
    
    def save_index(self, save_path: str):
        """Save index and metadata"""
        # If directory exists, create a new directory with incremental suffix
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(save_path, "vector_index.faiss"))
        
        # Save image paths
        with open(os.path.join(save_path, "image_paths.txt"), 'w', encoding='utf-8') as f:
            for path in self.image_paths:
                f.write(path + '\n')
        
        # Save embeddings (optional)
        np.save(os.path.join(save_path, "embeddings.npy"), self.embeddings)
        
        # Save model info
        with open(os.path.join(save_path, "model_info.txt"), 'w') as f:
            f.write(f"model: {self.model_name}\n")
            f.write(f"dimension: {self.dimension}\n")
        
        print(f"Index saved to {save_path}")
    
    def load_index(self, load_path: str):
        """Load index and metadata"""
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(load_path, "vector_index.faiss"))
        
        # Load image paths
        with open(os.path.join(load_path, "image_paths.txt"), 'r', encoding='utf-8') as f:
            self.image_paths = [line.strip() for line in f.readlines()]
        
        # Load embeddings (optional)
        embeddings_path = os.path.join(load_path, "embeddings.npy")
        if os.path.exists(embeddings_path):
            self.embeddings = np.load(embeddings_path)
        
        # Load model info
        model_info_path = os.path.join(load_path, "model_info.txt")
        if os.path.exists(model_info_path):
            with open(model_info_path, 'r') as f:
                for line in f:
                    if line.startswith('model:'):
                        self.model_name = line.split(':')[1].strip()
                    if line.startswith('dimension:'):
                        self.dimension = int(line.split(':')[1].strip())
        
        print(f"Index loaded successfully with {self.index.ntotal} vectors")


class ImageSearcher:
    """
    Search engine for finding similar images based on image embeddings.
    """
    def __init__(self, vector_db: VectorDatabase, embedder: ImageEmbedder):
        self.vector_db = vector_db
        self.embedder = embedder
    
    def search(self, query_image_path: str, top_k: int = 5) -> List[Dict]:
        """Search for similar images"""
        # Extract query image features
        query_features = self.embedder.extract_features(query_image_path)
        if query_features is None:
            return []
        
        query_features = query_features.astype('float32').reshape(1, -1)
        
        # Normalize query vector
        faiss.normalize_L2(query_features)
        
        # Search for similar vectors
        similarities, indices = self.vector_db.index.search(query_features, top_k)
        
        # Organize results
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.vector_db.image_paths):  # Ensure index validity
                results.append({
                    'rank': i + 1,
                    'image_path': self.vector_db.image_paths[idx],
                    'similarity': float(similarity)
                })
        
        return results
    
    def search_by_image(self, query_image: Image.Image, top_k: int = 5) -> List[Dict]:
        """Search directly using PIL Image object"""
        # Temporarily save image and search
        temp_path = "temp_query_image.jpg"
        query_image.save(temp_path)
        results = self.search(temp_path, top_k)
        
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return results


def create_results_image(query_path: str, results: List[Dict], save_path: Optional[str] = None, 
                        max_width: int = 1000) -> Image.Image:
    """Create combined image of query image and search results (optimized layout)"""
    # Load query image
    query_img = Image.open(query_path)
    
    # Resize query image - occupies 1/3 width on the left
    query_width, query_height = query_img.size
    query_display_width = max_width // 3
    query_display_height = int(query_display_width * query_height / query_width)
    query_img = query_img.resize((query_display_width, query_display_height), Image.Resampling.LANCZOS)
    
    # Load and resize result images
    result_imgs = []
    for result in results[:5]:  # Only take top 5 results
        img = Image.open(result['image_path'])
        img_width, img_height = img.size
        
        # Each result image occupies 1/4 of the remaining 2/3 width (i.e., 1/6 of total width)
        result_display_width = (max_width - query_display_width) // 4
        result_display_height = int(result_display_width * img_height / img_width)
        img = img.resize((result_display_width, result_display_height), Image.Resampling.LANCZOS)
        result_imgs.append((img, result['similarity'], result['rank']))
    
    # Calculate combined image height
    # Query image height + two rows of result images height + spacing and titles
    row_height = max([img.height for img, _, _ in result_imgs]) if result_imgs else 0
    total_height = max(query_display_height, row_height * 2) + 150  # Add space for titles and spacing
    
    # Create combined image
    merged_img = Image.new('RGB', (max_width, total_height), 'white')
    
    # Paste query image (left-centered)
    query_y = (total_height - query_display_height) // 2
    merged_img.paste(query_img, (20, query_y))
    
    # Paste result images (right in two rows)
    results_start_x = query_display_width + 40
    
    # First row: first 3 results
    for i in range(3):
        if i < len(result_imgs):
            img, similarity, rank = result_imgs[i]
            x = results_start_x + i * (result_imgs[0][0].width + 20)
            y = 40
            merged_img.paste(img, (x, y))
    
    # Second row: last 2 results (centered)
    for i in range(3, 5):
        if i < len(result_imgs):
            img, similarity, rank = result_imgs[i]
            # Center the last two images
            row_width = 2 * result_imgs[0][0].width + 20
            start_x = results_start_x + (max_width - results_start_x - row_width) // 2
            x = start_x + (i - 3) * (result_imgs[0][0].width + 20)
            y = 40 + row_height + 40
            merged_img.paste(img, (x, y))
    
    # Add text annotations
    draw = ImageDraw.Draw(merged_img)
    
    # Try to load fonts, fallback to default if unsuccessful
    try:
        font_large = ImageFont.truetype("arial.ttf", 24)
        font_medium = ImageFont.truetype("arial.ttf", 20)
        font_small = ImageFont.truetype("arial.ttf", 16)
    except:
        # Use default font
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Add query image title
    query_title = "Query Image"
    draw.text((query_display_width // 2 + 20, 30), query_title, fill='black', font=font_large, anchor='mm')
    
    # Add results title
    results_title = "Search Results"
    draw.text((results_start_x + (max_width - results_start_x) // 2, 30), results_title, 
              fill='black', font=font_large, anchor='mm')
    
    # Add ranking and similarity for result images
    # First row
    for i in range(3):
        if i < len(result_imgs):
            _, similarity, rank = result_imgs[i]
            x = results_start_x + i * (result_imgs[0][0].width + 20) + result_imgs[0][0].width // 2
            y = 40 + row_height + 10
            
            rank_text = f"Rank #{rank}"
            similarity_text = f"Similarity: {similarity:.3f}"
            
            draw.text((x, y), rank_text, fill='black', font=font_medium, anchor='mm')
            draw.text((x, y + 10), similarity_text, fill='black', font=font_small, anchor='mm')
    
    # Second row
    for i in range(3, 5):
        if i < len(result_imgs):
            _, similarity, rank = result_imgs[i]
            # Center the last two images
            row_width = 2 * result_imgs[0][0].width + 20
            start_x = results_start_x + (max_width - results_start_x - row_width) // 2
            x = start_x + (i - 3) * (result_imgs[0][0].width + 20) + result_imgs[0][0].width // 2
            y = 40 + row_height + 40 + row_height + 20
            
            rank_text = f"Rank #{rank}"
            similarity_text = f"Similarity: {similarity:.3f}"
            
            draw.text((x, y), rank_text, fill='black', font=font_medium, anchor='mm')
            draw.text((x, y + 15), similarity_text, fill='black', font=font_small, anchor='mm')
    
    # Save or return image
    if save_path:
        merged_img.save(save_path)
        print(f"Results image saved to: {save_path}")
    
    return merged_img


def display_results(query_path: str, results: List[Dict], figsize: Tuple[int, int] = (15, 10), 
                   save_path: Optional[str] = None):
    """Visualize search results"""
    # Create combined image
    merged_img = create_results_image(query_path, results, save_path)
    
    # Display image
    plt.figure(figsize=figsize)
    plt.imshow(merged_img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def create_web_searcher(index_path: str) -> ImageSearcher:
    """Create searcher for web service"""
    # Get model info from index
    model_info_path = os.path.join(index_path, "model_info.txt")
    model_name = "resnet50"  # Default model
    
    if os.path.exists(model_info_path):
        with open(model_info_path, 'r') as f:
            for line in f:
                if line.startswith('model:'):
                    model_name = line.split(':')[1].strip()
    
    embedder = ImageEmbedder(model_name=model_name)
    vector_db = VectorDatabase()
    vector_db.load_index(index_path)
    return ImageSearcher(vector_db, embedder)


def build_index(image_folder: str, index_save_path: str, model_name: str = 'resnet50', 
               use_optimized: bool = True) -> Tuple[VectorDatabase, ImageEmbedder]:
    """Build image index"""
    print(f"Initializing image embedder (model: {model_name})...")
    embedder = ImageEmbedder(model_name=model_name)
    index_save_dir = os.path.join(index_save_path, model_name)

    # Set feature dimension based on model
    if model_name.startswith('dinov2'):
        if model_name == 'dinov2_small':
            dimension = 384
        elif model_name == 'dinov2_base':
            dimension = 768
        elif model_name == 'dinov2_large':
            dimension = 1024
        else:  # dinov2_giant
            dimension = 1536
    elif model_name.startswith('nomic'):
        dimension = 768  # nomic-embed-vision-v1.5 feature dimension
    elif model_name.startswith('blip2'):
        dimension = 768  # BLIP2 image embedding dimension
    else:
        if model_name == 'resnet50' or model_name == 'resnet101':
            dimension = 2048
        else:  # resnet18
            dimension = 512
    
    print(f"Initializing vector database (dimension: {dimension}, model: {model_name})...")
    vector_db = VectorDatabase(dimension=dimension, model_name=model_name)
    
    # Build index
    if use_optimized:
        print("Using optimized index building...")
        vector_db.build_index_optimized(image_folder, embedder)
    else:
        print("Using standard index building...")
        vector_db.build_index(image_folder, embedder)
    
    # Save index and model info
    vector_db.save_index(index_save_dir)
    
    print("Index building completed!")
    
    return vector_db, embedder, index_save_dir


def search_image(query_image_path: str, index_path: str, top_k: int = 5, 
                show_results: bool = False, save_results: bool = False) -> List[Dict]:
    """Search for similar images"""
    # Get model info from index
    model_info_path = os.path.join(index_path, "model_info.txt")
    model_name = "resnet50"  # Default model
    
    if os.path.exists(model_info_path):
        with open(model_info_path, 'r') as f:
            for line in f:
                if line.startswith('model:'):
                    model_name = line.split(':')[1].strip()
    
    print(f"Using model: {model_name}")
    
    # Initialize components
    embedder = ImageEmbedder(model_name=model_name)
    vector_db = VectorDatabase()
    vector_db.load_index(index_path)
    searcher = ImageSearcher(vector_db, embedder)
    
    # Perform search
    if os.path.exists(query_image_path):
        results = searcher.search(query_image_path, top_k=top_k)
        
        print("\nSearch results:")
        for result in results:
            print(f"Rank: {result['rank']}, Similarity: {result['similarity']:.4f}")
            print(f"Image path: {result['image_path']}\n")
        
        # Save results image
        if save_results and results:
            # Create results directory
            # results_dir = "outputs/search_results"
            results_dir = os.path.join(save_path, model_name)
            os.makedirs(results_dir, exist_ok=True)
            
            # Generate filename
            query_name = os.path.splitext(os.path.basename(query_image_path))[0]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(results_dir, f"result_{query_name}_{timestamp}.jpg")
            
            # Create and save results image
            merged_img = create_results_image(query_image_path, results, save_path=save_path)
        elif results:
            merged_img = create_results_image(query_image_path, results)
        # Visualize results
        if show_results and results:
            display_results(query_image_path, results)
        
        return results, merged_img, model_name
    else:
        print(f"Query image does not exist: {query_image_path}")
        return []



def find_similar_image_groups_from_folder(image_folder: str, model_name: str = 'resnet50', 
                                        images_per_group: int = 4, save_dir: str = "similar_groups",
                                        min_cluster_size: Optional[int] = None) -> List[List[str]]:
    """
    Find similar image groups by clustering image embeddings from a folder using the improved workflow:
    Feature Extraction → L2 Normalization → PCA Dimensionality Reduction → HDBSCAN Clustering
    
    Args:
        image_folder: Path to the folder containing images
        model_name: Name of the model to use for embedding extraction
        images_per_group: Number of images to include in each group visualization
        save_dir: Directory to save group visualization results
        min_cluster_size: Minimum cluster size for HDBSCAN. If None, automatically determined.
    
    Returns:
        List of image groups, where each group is a list of image paths
    """
    print(f"Finding similar image groups in folder: {image_folder}")
    print(f"Using model: {model_name}")
    
    # Get all image files from the folder
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(root, file))

    if not image_files:
        print("No images found in the specified folder")
        return []
    
    print(f"Found {len(image_files)} images in folder")
    
    # Create embedder
    embedder = ImageEmbedder(model_name=model_name)
    
    # Extract features for all images
    print("Extracting image features...")
    embeddings = []
    valid_image_paths = []
    
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            features = embedder.extract_features(image_path)
            if features is not None:
                embeddings.append(features)
                valid_image_paths.append(image_path)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    if len(embeddings) < 2:
        print("Not enough valid images for clustering")
        return []
    
    embeddings = np.array(embeddings)
    print(f"Successfully processed {len(embeddings)} images")
    
    # Step 1: L2 Normalization
    print("Performing L2 normalization...")
    faiss.normalize_L2(embeddings)
    
    # Step 2: PCA Dimensionality Reduction (if needed)
    print("Performing PCA dimensionality reduction...")
    reduced_embeddings = _pca_reduce_dimensions(embeddings)
    
    # Step 3: HDBSCAN Clustering
    # print("Performing HDBSCAN clustering...")
    # cluster_labels = _hdbscan_clustering(reduced_embeddings, min_cluster_size)
    print("Performing K-means clustering...")
    n_clusters = max(2, min(20, len(embeddings) // 5))  # Heuristic for cluster count
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Group images by cluster
    unique_labels = np.unique(cluster_labels)
    unique_labels = unique_labels[unique_labels != -1]  # Exclude noise points
    
    image_groups = []
    for label in unique_labels:
        indices = np.where(cluster_labels == label)[0]
        group_images = [valid_image_paths[i] for i in indices]
        if len(group_images) >= 2:  # Only include groups with at least 2 images
            image_groups.append(group_images)
    
    # Sort groups by size (largest first)
    image_groups.sort(key=len, reverse=True)
    
    if not image_groups:
        print("No meaningful clusters found")
        return []
    
    # Create results directory with model name subdirectory
    model_save_dir = os.path.join(save_dir, model_name)
    # If directory exists, create a new directory with incremental suffix
    if os.path.exists(model_save_dir):
        i = 2
        while os.path.exists(f"{model_save_dir}_{i}"):
            i += 1
        model_save_dir = f"{model_save_dir}_{i}"

    os.makedirs(model_save_dir, exist_ok=True)
    
    # Process each group to find most similar images and create visualizations
    result_groups = []
    
    for group_idx, group_images in enumerate(image_groups):
        print(f"Processing group {group_idx+1} with {len(group_images)} images...")
        
        # For small groups, use all images
        if len(group_images) <= images_per_group:
            selected_images = group_images
        else:
            # For larger groups, find the most representative images
            group_embeddings = []
            for img_path in group_images:
                idx = valid_image_paths.index(img_path)
                group_embeddings.append(reduced_embeddings[idx])
            
            group_embeddings = np.array(group_embeddings)
            
            # Calculate centroid of the group
            centroid = np.mean(group_embeddings, axis=0, keepdims=True)
            faiss.normalize_L2(centroid)
            
            # Find images closest to centroid
            similarities = np.dot(group_embeddings, centroid.T).flatten()
            top_indices = np.argsort(similarities)[::-1][:images_per_group]
            selected_images = [group_images[i] for i in top_indices]
        
        result_groups.append(selected_images)

        # Save group images
    if result_groups:
        print(f"Found {len(result_groups)} groups of similar images")
        collage_paths = save_group_images(result_groups, model_save_dir)
        print(f"Saved {len(collage_paths)} group collages")
    else:
        print("No similar image groups found")

        
        
    print(f"Found {len(result_groups)} similar image groups")
    return result_groups, collage_paths


def _pca_reduce_dimensions(embeddings: np.ndarray, target_dim: int = 128) -> np.ndarray:
    """
    Reduce dimensionality of embeddings using PCA.
    
    Args:
        embeddings: Input embeddings matrix (n_samples, n_features)
        target_dim: Target dimension after reduction
    
    Returns:
        Reduced dimensionality embeddings
    """
    from sklearn.decomposition import PCA
    
    # If already low-dimensional, no need to reduce
    if embeddings.shape[1] <= target_dim:
        return embeddings
    
    # Adjust target_dim to be at most min(n_samples, n_features) - 1
    max_components = min(embeddings.shape) - 1
    adjusted_target_dim = min(target_dim, max_components)
    
    if adjusted_target_dim <= 0:
        print(f"Not enough samples ({embeddings.shape[0]}) for PCA, returning original embeddings")
        return embeddings

    # Apply PCA
    pca = PCA(n_components=adjusted_target_dim)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    
    print(f"Reduced dimensions from {embeddings.shape[1]} to {reduced_embeddings.shape[1]} "
          f"(explained variance ratio: {np.sum(pca.explained_variance_ratio_):.3f})")
    
    return reduced_embeddings


def _hdbscan_clustering(embeddings: np.ndarray, min_cluster_size: Optional[int] = None) -> np.ndarray:
    """
    Perform HDBSCAN clustering on embeddings.
    
    Args:
        embeddings: Input embeddings matrix (n_samples, n_features)
        min_cluster_size: Minimum cluster size. If None, automatically determined.
    
    Returns:
        Cluster labels for each sample (-1 indicates noise)
    """
    try:
        from hdbscan import HDBSCAN
    except ImportError:
        print("HDBSCAN not available. Please install hdbscan package (pip install hdbscan)")
        # Fallback to simple KMeans clustering
        # from sklearn.cluster import KMeans
        n_clusters = max(2, min(20, len(embeddings) // 5))  # Heuristic for cluster count
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(embeddings)
    
    # Automatically determine min_cluster_size if not provided
    if min_cluster_size is None:
        # Heuristic: at least 2 samples per cluster, but not more than 10% of total samples
        min_cluster_size = max(2, min(20, len(embeddings) // 10))
        print(f"Auto-determined min_cluster_size: {min_cluster_size}")
    
    # Apply HDBSCAN
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
    cluster_labels = clusterer.fit_predict(embeddings)
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"HDBSCAN found {n_clusters} clusters with {n_noise} noise points")
    
    return cluster_labels

def create_group_collage(group_images, group_idx, output_dir, max_width=1200):
    """
    Create a collage of images in a group
    
    :param group_images: List of image paths in the group
    :param group_idx: Group index for naming
    :param output_dir: Directory to save the collage
    :param max_width: Maximum width of the collage
    :return: Path to the saved collage
    """
    if not group_images:
        return None
    
    # Load and resize images
    images = []
    for img_path in group_images:
        try:
            img = Image.open(img_path)
            images.append(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    
    if not images:
        return None
    
    # Determine layout (up to 2 rows)
    num_images = len(images)
    if num_images <= 3:
        cols = num_images
        rows = 1
    else:
        cols = min(3, (num_images + 1) // 2)
        rows = 2
    
    # Resize images to fit in grid
    target_width = max_width // cols
    resized_images = []
    max_heights = [0] * rows  # Track max height per row
    
    for i, img in enumerate(images):
        # Calculate row and column
        row = i // cols
        col = i % cols
        
        # Resize image
        aspect_ratio = img.height / img.width
        new_height = int(target_width * aspect_ratio)
        resized_img = img.resize((target_width, new_height), Image.Resampling.LANCZOS)
        resized_images.append(resized_img)
        
        # Update max height for row
        if new_height > max_heights[row]:
            max_heights[row] = new_height
    
    # Create collage
    total_height = sum(max_heights) + 20 * (rows + 1)  # Add padding
    collage = Image.new('RGB', (max_width, total_height), 'white')
    
    # Paste images
    y_offset = 20
    for row in range(rows):
        x_offset = 0
        start_idx = row * cols
        end_idx = min(start_idx + cols, len(resized_images))
        
        for i in range(start_idx, end_idx):
            img = resized_images[i]
            collage.paste(img, (x_offset, y_offset))
            x_offset += target_width
        
        y_offset += max_heights[row] + 20
    
    # Save collage
    collage_path = os.path.join(output_dir, f"group_{group_idx}_collage.jpg")
    collage.save(collage_path, "JPEG", quality=85)
    return collage_path

def save_group_images(groups, output_dir):
    """
    Save groups of images as individual files and collages
    
    :param groups: List of image groups (each group is a list of image paths)
    :param output_dir: Directory to save group images
    :return: List of saved collage paths
    """
    # group_output_dir = os.path.join(output_dir, "similar_groups")
    
    os.makedirs(output_dir, exist_ok=True)
    
    collage_paths = []
    
    for idx, group in enumerate(groups):
        # Create directory for this group
        group_dir = os.path.join(output_dir, f"group_{idx+1}")
        os.makedirs(group_dir, exist_ok=True)
        
        # Copy individual images to group directory
        for img_idx, img_path in enumerate(group):
            try:
                img_name = os.path.basename(img_path)
                dest_path = os.path.join(group_dir, f"{img_idx+1}_{img_name}")
                img = Image.open(img_path)
                img.save(dest_path)
            except Exception as e:
                print(f"Error copying image {img_path}: {e}")
        
        # Create and save collage for the group
        collage_path = create_group_collage(group, idx+1, output_dir)
        if collage_path:
            collage_paths.append(collage_path)
            print(f"Saved group {idx+1} collage to: {collage_path}")
    
    return collage_paths

# Update the usage example in the main section:
if __name__ == "__main__":
    # Example usage
    # Build index
    # vector_db, embedder = build_index("./images", "./index", model_name="resnet50")
    
    # Search
    # results = search_image("./query.jpg", "./index")
    
    # Find similar groups with automatic clustering
    # groups = find_similar_image_groups_from_folder("./images", model_name="resnet50")
    
    print("This module can be imported and used in other scripts.")
    print("Available functions:")
    print("- build_index(image_folder, index_save_path, model_name, use_optimized)")
    print("- search_image(query_image_path, index_path, top_k, show_results, save_results)")
    print("- find_similar_image_groups_from_folder(image_folder, model_name, images_per_group, save_dir, min_cluster_size)")
    print("- create_web_searcher(index_path)")
    print("- ImageEmbedder, VectorDatabase, ImageSearcher classes for advanced usage")