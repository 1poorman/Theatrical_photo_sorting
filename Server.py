# myserver.py
import os
import cv2
import torch
import sys
import datetime
sys.path.append(".")
from flask import Flask, request, jsonify, render_template_string, send_file
from werkzeug.utils import secure_filename
import tempfile
import shutil
import json
from PIL import Image
from detection.PersonMaskCreator import PersonMaskCreator
from inpaint.inpaint_lama_ds import ImageInpainter
# from seg_clothes.ClothesSegment import ClothesSegmenter
from seg_clothes.yolo_seg import PersonesSegmenter
from embedding.image_search_system_module import (
    build_index, search_image, find_similar_image_groups_from_folder
)
from shot.views_classify import ShotTypeClassifier
# sys.path.append(os.path.join(os.path.dirname(__file__), "face_recognition"))
from face_recognition.face_recognition import FaceRecognitionSystem
import threading
import time

# Add a global dictionary to track progress
progress_tracker = {}

sys.path.append(os.getcwd())
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Max 100MB upload

# Global model instances
face_recognition_model = None 
detection_model = None
# segclothes_model = None
segpersones_model = None
inpainter_model = None
pose_model = None

device = "cuda: 1" if torch.cuda.is_available() else "cpu"

# Default paths - these should be configured based on your environment
DEFAULT_DETECTION_MODEL_PATH = "./weights/rtdetr-x.pt"
# DEFAULT_SEGCLOTHES_MODEL_PATH = "./weights/segformer_b2_clothes"
DEFAULT_SEGPERSONES_MODEL_PATH = "./weights/yolo11x-seg.pt"
DEFAULT_INPAINTER_MODEL_PATH = "./weights/cv_fft_inpainting_lama"
DEFAULT_POSE_MODEL_PATH = "./weights/yolo11l-pose.pt"
DEFAULT_OUTPUT_DIR = "./outputs"
DEFAULT_SCRFD_MODEL_PATH = "./weights/scrfd/scrfd_10g_bnkps.onnx"
DEFAULT_ARCFACE_MODEL_PATH = "./weights/arcface/Glint100.onnx"
DEFAULT_FACE_DATABASE_PATH = "./outputs/face_index"  # Default path for face database

# Create directories if they don't exist
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
TEMP_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Store the last built index path
last_index_path = None

# Function to clean up temp directory
def cleanup_temp_directory():
    """
    Clean up old files in the temp directory
    """
    try:
        # Remove all files and folders in temp directory older than 1 hour
        current_time = time.time()
        for item in os.listdir(TEMP_DIR):
            item_path = os.path.join(TEMP_DIR, item)
            # Get the modification time of the item
            mod_time = os.path.getmtime(item_path)
            # If older than 1 hour (3600 seconds), remove it
            if current_time - mod_time > 3600:
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
        
        # Also ensure we don't have too many items (e.g., more than 100)
        items = os.listdir(TEMP_DIR)
        if len(items) > 100:
            # Sort by modification time and remove oldest items
            items_with_time = [(item, os.path.getmtime(os.path.join(TEMP_DIR, item))) for item in items]
            items_with_time.sort(key=lambda x: x[1])  # Sort by time (oldest first)
            
            # Remove the oldest items to keep only 50
            items_to_remove = items_with_time[:len(items)-50]
            for item, _ in items_to_remove:
                item_path = os.path.join(TEMP_DIR, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    
    except Exception as e:
        print(f"Warning: Error during temp directory cleanup: {e}")

# Enhanced function to clean up temp directory before creating new temp folder
def prepare_temp_directory():
    """
    Prepare temp directory by cleaning up old files and returning a new temp folder path
    """
    # Clean up old files first
    cleanup_temp_directory()
    
    # Create a new temporary directory
    temp_dir = tempfile.mkdtemp(dir=TEMP_DIR)
    return temp_dir

# Add a periodic cleanup function
def periodic_cleanup():
    """
    Periodically clean up temp directory (runs every 30 minutes)
    """
    while True:
        time.sleep(1800)  # Sleep for 30 minutes
        cleanup_temp_directory()

# Start periodic cleanup in a separate thread
# cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
# cleanup_thread.start()
# Add a progress tracking decorator
def track_progress(func):
    def wrapper(*args, **kwargs):
        # Generate a unique task ID
        task_id = str(time.time())
        progress_tracker[task_id] = {"status": "started", "progress": 0, "message": "Starting..."}
        
        # Run the function in a separate thread
        def run_task():
            try:
                progress_tracker[task_id]["status"] = "running"
                result = func(*args, **kwargs, progress_callback=lambda p, m: update_progress(task_id, p, m))
                progress_tracker[task_id]["status"] = "completed"
                progress_tracker[task_id]["result"] = result
            except Exception as e:
                progress_tracker[task_id]["status"] = "error"
                progress_tracker[task_id]["error"] = str(e)
        
        thread = threading.Thread(target=run_task)
        thread.start()
        
        return task_id
    return wrapper

def update_progress(task_id, progress, message):
    if task_id in progress_tracker:
        progress_tracker[task_id]["progress"] = progress
        progress_tracker[task_id]["message"] = message

def initialize_models(detection_model_path=DEFAULT_DETECTION_MODEL_PATH,
                    #  segclothes_model_path=DEFAULT_SEGCLOTHES_MODEL_PATH,
                     segpersones_model_path=DEFAULT_SEGPERSONES_MODEL_PATH,
                     inpainter_model_path=DEFAULT_INPAINTER_MODEL_PATH,
                     pose_model_path=DEFAULT_POSE_MODEL_PATH,
                     scrfd_model_path=DEFAULT_SCRFD_MODEL_PATH,
                     arcface_model_path=DEFAULT_ARCFACE_MODEL_PATH):
    """
    Initialize all models
    """
    global detection_model, segpersones_model, inpainter_model, pose_model, face_recognition_model 
    
    try:
        # Check CUDA availability and set device
        if torch.cuda.is_available():
            device = "cuda:1"  # Use cuda:0 instead of cuda:1
            print(f"CUDA is available. Using device: {device}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
        else:
            device = "cpu"
            print("CUDA is not available. Using CPU.")

        if os.path.exists(scrfd_model_path) and os.path.exists(arcface_model_path):
            try:
                face_recognition_model = FaceRecognitionSystem(scrfd_model_path, arcface_model_path, device=device)
                print(f"Face recognition model loaded")
            except Exception as e:
                print(f"Failed to load face recognition model: {e}")
        else:
            print(f"Face recognition models not found at {scrfd_model_path} or {arcface_model_path}")
            
        if detection_model_path and os.path.exists(detection_model_path):
            detection_model = PersonMaskCreator(detection_model_path)
            print(f"Detection model loaded from {detection_model_path}")
        else:
            print(f"Detection model not found at {detection_model_path}")
        
        # if segclothes_model_path and os.path.exists(segclothes_model_path):
        #     segclothes_model = ClothesSegmenter(segclothes_model_path)
        #     print(f"Segmentation model loaded from {segclothes_model_path}")
        # else:
        #     print(f"Segmentation model not found at {segclothes_model_path}")

        if segpersones_model_path and os.path.exists(segpersones_model_path):
            segpersones_model = PersonesSegmenter(segpersones_model_path)
            print(f"Segmentation model loaded from {segpersones_model_path}")
        else:
            print(f"Segmentation model not found at {segpersones_model_path}")
            
        if inpainter_model_path and os.path.exists(inpainter_model_path):
            inpainter_model = ImageInpainter(inpainter_model_path, max_size=1024)
            print(f"Inpainting model loaded from {inpainter_model_path}")
        else:
            print(f"Inpainting model not found at {inpainter_model_path}")
        
        if pose_model_path and os.path.exists(pose_model_path):
            pose_model = ShotTypeClassifier(pose_model_path)
            print(f"Pose model loaded from {pose_model_path}")
        else:
            print(f"Pose model not found at {pose_model_path}")
            
        return True
    except Exception as e:
        print(f"Error initializing models: {e}")
        return False

def get_error(code=-1, message=""):
    return {"code": code, "message": message}

def serialize_results(results):
    """
    Convert YOLO results object to JSON serializable format
    """
    if isinstance(results, list):
        serialized = []
        for result in results:
            if hasattr(result, 'boxes'):
                boxes_data = []
                if result.boxes is not None:
                    for box in result.boxes:
                        box_data = {
                            'xyxy': box.xyxy.cpu().numpy().tolist() if hasattr(box.xyxy, 'cpu') else box.xyxy.tolist(),
                            'conf': box.conf.cpu().item() if hasattr(box.conf, 'cpu') else box.conf.item(),
                            'cls': box.cls.cpu().item() if hasattr(box.cls, 'cpu') else box.cls.item()
                        }
                        boxes_data.append(box_data)
                serialized.append({'boxes': boxes_data})
            else:
                serialized.append(str(result))
        return serialized
    else:
        # Handle single result object
        if hasattr(results, 'boxes'):
            boxes_data = []
            if results.boxes is not None:
                for box in results.boxes:
                    box_data = {
                        'xyxy': box.xyxy.cpu().numpy().tolist() if hasattr(box.xyxy, 'cpu') else box.xyxy.tolist(),
                        'conf': box.conf.cpu().item() if hasattr(box.conf, 'cpu') else box.conf.item(),
                        'cls': box.cls.cpu().item() if hasattr(box.cls, 'cpu') else box.cls.item()
                    }
                    boxes_data.append(box_data)
            return {'boxes': boxes_data}
        else:
            return str(results)

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Unhandled exception: {str(e)}")
    return jsonify(get_error(message=str(e))), 500

@app.route("/", methods=["GET"])
def index():
    """
    Main UI page with forms for all functionalities
    """
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>Image Processing Toolkit</title>
    <meta charset="UTF-8">
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background-color: #f5f5f5;
        }
        h1, h2 { 
            color: #333; 
            text-align: center;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .section {
            background: #fafafa;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
            border-left: 4px solid #007bff;
        }
        .form-group {
            margin: 10px 0;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input[type="file"], input[type="text"], select {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
        }
        button {
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin: 5px 0;
        }
        button:hover {
            background-color: #0056b3;
        }
        .result {
            margin-top: 15px;
            padding: 10px;
            background: #e9f7ef;
            border-radius: 4px;
            display: none;
        }
        .error {
            background: #f8d7da;
            color: #721c24;
        }
        .success {
            background: #d4edda;
            color: #155724;
        }
        img {
            max-width: 100%;
            height: auto;
            margin-top: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        pre {
            background: #f8f9fa;
            padding: 10px;
            overflow-x: auto;
            white-space: pre-wrap;
        }
        .image-container {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }
        .image-item {
            flex: 1 1 300px;
            text-align: center;
        }
        .image-item img {
            max-width: 100%;
            height: auto;
        }
        .image-item p {
            font-weight: bold;
            margin: 5px 0;
        }
        /* Spinner animation */
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 2s linear infinite;
            margin: 0 auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 Image Processing Toolkit</h1>
        <p style="text-align:center;">A comprehensive tool for theatrical photo processing and organization</p>
        
        <!-- Health Check -->
        <div class="section">
            <h2>🏥 System Status</h2>
            <button onclick="checkHealth()">Check System Health</button>
            <div id="health-result" class="result"></div>
        </div>
        
        <!-- Build Face Database -->
        <div class="section">
            <h2>👤 Build Face Database</h2>
            <form id="build-face-db-form">
                <div class="form-group">
                    <label for="face-db-folder">Face Database Folder Path:</label>
                    <input type="text" id="face-db-folder" name="face_db_folder" placeholder="/path/to/face/database/folder" required>
                </div>
                <button type="submit">Build Face Database</button>
            </form>
            <div id="build-face-db-result" class="result"></div>
        </div>
        
        <!-- Recognize Faces -->
        <div class="section">
            <h2>👁️ Recognize Faces</h2>
            <form id="recognize-faces-form" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="recognize-image">Upload Image:</label>
                    <input type="file" id="recognize-image" name="image" accept="image/*" required>
                </div>
                <button type="submit">Recognize Faces</button>
            </form>
            <div id="recognize-faces-result" class="result"></div>
        </div>
        
        <!-- Combined Person Processing -->
        <div class="section">
            <h2>👤 Person Detection, Segmentation & Inpainting</h2>
            <form id="combined-form" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="combined-image">Upload Image:</label>
                    <input type="file" id="combined-image" name="image" accept="image/*" required>
                </div>
                <input type="hidden" id="combined-detection-model-path" name="detection_model_path" value="{{ detection_model_path }}">
                <input type="hidden" id="combined-segpersones_model_path" name="segpersones_model_path" value="{{ segpersones_model_path }}">
                <input type="hidden" id="combined-inpainter-model-path" name="inpainter_model_path" value="{{ inpainter_model_path }}">
                <button type="submit">Process Image</button>
            </form>
            <div id="combined-result" class="result"></div>
        </div>
        
        <!-- Build Embedding Index -->
        <div class="section">
            <h2>📚 Build Image Index</h2>
            <form id="index-form">
                <div class="form-group">
                    <label for="image-folder">Image Folder Path:</label>
                    <input type="text" id="image-folder" name="image_folder" placeholder="/path/to/image/folder" required>
                </div>
                <div class="form-group">
                    <label for="index-save-path">Index Save Path:</label>
                    <input type="text" id="index-save-path" name="index_save_path" placeholder="/path/to/save/index" required>
                </div>
                <div class="form-group">
                    <label for="index-model-name">Model Name:</label>
                    <select id="index-model-name" name="model_name">
                        <option value="resnet50">ResNet50</option>
                        <option value="resnet101">ResNet101</option>
                        <option value="dinov2_small">DINOv2 Small</option>
                        <option value="dinov2_base">DINOv2 Base</option>
                        <option value="dinov2_large">DINOv2 Large</option>                        
                    </select>
                </div>
                <button type="submit">Build Index</button>
            </form>
            <div id="index-result" class="result"></div>
        </div>
        
        <!-- Search Similar Images -->
        <div class="section">
            <h2>🔍 Search Similar Images</h2>
            <form id="search-form" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="search-query-image">Query Image:</label>
                    <input type="file" id="search-query-image" name="query_image" accept="image/*" required>
                </div>
                <div class="form-group">
                    <label for="search-index-path">Index Path (leave empty to use last built index):</label>
                    <input type="text" id="search-index-path" name="index_path" placeholder="/path/to/index">
                </div>
                <div class="form-group">
                    <label for="search-top-k">Number of Results:</label>
                    <input type="number" id="search-top-k" name="top_k" value="5" min="1" max="20">
                </div>
                <button type="submit">Search Images</button>
            </form>
            <div id="search-result" class="result"></div>
        </div>
        
        <!-- Group Similar Images -->
        <div class="section">
            <h2>ParallelGroup Similar Images</h2>
            <form id="group-form">
                <div class="form-group">
                    <label for="group-image-folder">Image Folder Path:</label>
                    <input type="text" id="group-image-folder" name="image_folder" placeholder="/path/to/image/folder" required>
                </div>
                <div class="form-group">
                    <label for="group-model-name">Model Name:</label>
                    <select id="group-model-name" name="model_name">
                        <option value="resnet50">ResNet50</option>
                        <option value="resnet101">ResNet101</option>
                        <option value="dinov2_small">DINOv2 Small</option>
                        <option value="dinov2_base">DINOv2 Base</option>
                        <option value="dinov2_large">DINOv2 Large</option>                        
                    </select>
                </div>
                <div class="form-group">
                    <label for="save-dir">Save Directory:</label>
                    <input type="text" id="save-dir" name="save_dir" placeholder="similar_groups">
                </div>
                <button type="submit">Group Images</button>
            </form>
            <div id="group-result" class="result"></div>
        </div>
    </div>

    <script>
        // Helper function to clear result area
        function clearResult(elementId) {
            const element = document.getElementById(elementId);
            element.innerHTML = '';
            element.style.display = 'none';
            element.className = 'result';
        }
        
        // Helper function to show loading indicator
        function showLoading(elementId, message) {
            const element = document.getElementById(elementId);
            element.innerHTML = `
                <div style="text-align: center; padding: 20px;">
                    <div class="spinner"></div>
                    <p style="margin-top: 10px;">${message || 'Processing...'}</p>
                </div>
            `;
            element.style.display = 'block';
            element.className = 'result';
        }

        // Helper function to display results
        function displayResult(elementId, data, isError = false) {
            const element = document.getElementById(elementId);
            element.innerHTML = '';
            element.style.display = 'block';
            
            if (isError) {
                element.className = 'result error';
                element.innerHTML = `<strong>Error:</strong> ${data.message || data}`;
            } else {
                element.className = 'result success';
                if (typeof data === 'object') {
                    if (data.data && data.data.detect_path && data.data.extracted_path && data.data.inpainted_path) {
                        element.innerHTML = `
                            <strong>Success!</strong>
                            <p>All processing steps completed successfully.</p>
                            <div class="image-container">
                                <div class="image-item">
                                    <p>Original Image</p>
                                    <img src="/api/file?path=${encodeURIComponent(data.data.original_path)}" alt="Original">
                                </div>
                                <div class="image-item">
                                    <p>Detect Image</p>
                                    <img src="/api/file?path=${encodeURIComponent(data.data.detect_path)}" alt="Detect">
                                </div>
                                <div class="image-item">
                                    <p>Extracted Clothing</p>
                                    <img src="/api/file?path=${encodeURIComponent(data.data.extracted_path)}" alt="Extracted">
                                </div>
                                <div class="image-item">
                                    <p>Pose Estimation</p>
                                    <img src="/api/file?path=${encodeURIComponent(data.data.pose_path)}" alt="Pose">
                                </div>
                                <div class="image-item">
                                    <p>Inpainted Image</p>
                                    <img src="/api/file?path=${encodeURIComponent(data.data.inpainted_path)}" alt="Inpainted">
                                </div>
                            </div>
                            <div style="background: #e3f2fd; padding: 10px; border-radius: 4px; margin: 10px 0;">
                                <strong>📸 Shot Type Classification:</strong><br>
                                Type: ${data.data.shot_type}<br>
                                Area_ratio: ${data.data.area_ratio || 'N/A'}
                            </div>
                            <p>Files saved at:</p>
                            <ul>
                                <li>Detect: ${data.data.detect_path}</li>
                                <li>Extracted: ${data.data.extracted_path}</li>
                                <li>Pose: ${data.data.pose_path}</li>
                                <li>Inpainted: ${data.data.inpainted_path}</li>                                
                            </ul>
                        `;
                    } else if (data.data && data.data.index_path) {
                        element.innerHTML = `
                            <strong>Success!</strong>
                            <p>Index built successfully with ${data.data.image_count} images.</p>
                            <p>Index saved at: ${data.data.index_path}</p>
                        `;
                    } else if (data.data && data.data.results) {
                        let resultsHtml = '<strong>Success! Search Results:</strong>';
                        if (data.data.results.length > 0) {
                            // Display combined result image if available
                            if (data.data.combined_result_path) {
                                resultsHtml += `
                                    <div class="image-item">
                                        <p>Combined Results</p>
                                        <img src="/api/file?path=${encodeURIComponent(data.data.combined_result_path)}" alt="Combined Results">
                                    </div>
                                `;
                            }
                            
                            // List similar image names
                            resultsHtml += '<p><strong>Similar images:</strong></p><ul>';
                            data.data.similar_image_names.forEach((name, index) => {
                                resultsHtml += `<li>Result #${index + 1}: ${name}</li>`;
                            });
                            resultsHtml += '</ul>';
                        } else {
                            resultsHtml += '<p>No similar images found.</p>';
                        }
                        element.innerHTML = resultsHtml;
                    } else if (data.data && data.data.groups) {
                        let resultsHtml = `<strong>Success!</strong>`;
                        resultsHtml += `<p>Found ${data.data.group_count} groups of similar images.</p>`;
                        
                        // Display collage images
                        if (data.data.collage_paths && data.data.collage_paths.length > 0) {
                            resultsHtml += `<p><strong>Group Collages:</strong></p>`;
                            resultsHtml += `<div class="image-container">`;
                            data.data.collage_paths.forEach((path, index) => {
                                resultsHtml += `
                                    <div class="image-item">
                                        <p>Group ${index + 1} Collage</p>
                                        <img src="/api/file?path=${encodeURIComponent(path)}" alt="Group ${index + 1} Collage">
                                    </div>
                                `;
                            });
                            resultsHtml += `</div>`;
                        }

                        // Display group information
                        data.data.groups.forEach(group => {
                            resultsHtml += `<p><strong>Group ${group.group_id}:</strong> ${group.image_count} images</p>`;
                            resultsHtml += `<ul>`;
                            group.image_names.forEach(name => {
                                resultsHtml += `<li>${name}</li>`;
                            });
                            resultsHtml += `</ul>`;
                        });
                        
                        
                        element.innerHTML = resultsHtml;
                    } else if (data.data && data.data.hasOwnProperty('face_db_built')) {
                        element.innerHTML = `
                            <strong>Success!</strong>
                            <p>Face database built successfully with ${data.data.person_count} persons.</p>
                        `;
                    } else if (data.data && data.data.hasOwnProperty('recognized_faces')) {
                        let resultsHtml = `<strong>Success!</strong>`;
                        resultsHtml += `<p>Found ${data.data.recognized_faces.length} faces in the image.</p>`;
                        if (data.data.annotated_image_path) {
                            resultsHtml += `
                                <div class="image-item">
                                    <p>Annotated Image</p>
                                    <img src="/api/file?path=${encodeURIComponent(data.data.annotated_image_path)}" alt="Annotated Image">
                                </div>
                            `;
                        }

                        if (data.data.recognized_faces.length > 0) {
                            resultsHtml += `<div class="image-container">`;
                            data.data.recognized_faces.forEach((face, index) => {
                                resultsHtml += `
                                    <div class="image-item">
                                        <p>Face ${index + 1}</p>
                                        <img src="/api/file?path=${encodeURIComponent(face.face_image_path)}" alt="Face ${index + 1}">
                                        <p><strong>${face.identified_as}</strong><br>Confidence: ${face.identification_confidence.toFixed(3)}</p>
                                    </div>
                                `;
                            });
                            resultsHtml += `</div>`;
                        }
                        
                        element.innerHTML = resultsHtml;
                    } else {
                        element.innerHTML = `<strong>Success!</strong><pre>${JSON.stringify(data, null, 2)}</pre>`;
                    }
                } else {
                    element.innerHTML = `<strong>Success!</strong> ${data}`;
                }
            }
        }

        // Health check
        function checkHealth() {
            clearResult('health-result');
            fetch('/api/health')
                .then(response => response.json())
                .then(data => {
                    displayResult('health-result', data);
                })
                .catch(error => {
                    displayResult('health-result', error, true);
                });
        }

        // Build face database form submission
        document.getElementById('build-face-db-form').addEventListener('submit', function(e) {
            e.preventDefault();
            clearResult('build-face-db-result');
            showLoading('build-face-db-result', 'Building face database...');
            
            const formData = new FormData(this);
            
            fetch('/api/face/build_database', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                displayResult('build-face-db-result', data, data.code !== 200);
            })
            .catch(error => {
                displayResult('build-face-db-result', error, true);
            });
        });
        
        // Recognize faces form submission
        document.getElementById('recognize-faces-form').addEventListener('submit', function(e) {
            e.preventDefault();
            clearResult('recognize-faces-result');
            showLoading('recognize-faces-result', 'Recognizing faces...');
            
            const formData = new FormData(this);
            
            fetch('/api/face/recognize', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                displayResult('recognize-faces-result', data, data.code !== 200);
            })
            .catch(error => {
                displayResult('recognize-faces-result', error, true);
            });
        });

        // Combined processing form submission
        document.getElementById('combined-form').addEventListener('submit', function(e) {
            e.preventDefault();
            clearResult('combined-result');
            showLoading('combined-result', 'Processing image...');
            
            const formData = new FormData(this);
            
            fetch('/api/image/process', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                displayResult('combined-result', data, data.code !== 200);
            })
            .catch(error => {
                displayResult('combined-result', error, true);
            });
        });

        // Build index form submission
        document.getElementById('index-form').addEventListener('submit', function(e) {
            e.preventDefault();
            clearResult('index-result');
            showLoading('index-result', 'Building image index...');
            
            const formData = new FormData(this);
            
            fetch('/api/embedding/build_index', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                displayResult('index-result', data, data.code !== 200);
            })
            .catch(error => {
                displayResult('index-result', error, true);
            });
        });

        // Search similar images form submission
        document.getElementById('search-form').addEventListener('submit', function(e) {
            e.preventDefault();
            clearResult('search-result');
            showLoading('search-result', 'Searching similar images...');
            
            const formData = new FormData(this);
            
            fetch('/api/embedding/search', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                displayResult('search-result', data, data.code !== 200);
            })
            .catch(error => {
                displayResult('search-result', error, true);
            });
        });

        // Group similar images form submission
        document.getElementById('group-form').addEventListener('submit', function(e) {
            e.preventDefault();
            clearResult('group-result');
            showLoading('group-result', 'Grouping similar images...');
            
            const formData = new FormData(this);
            
            fetch('/api/embedding/group_similar', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                displayResult('group-result', data, data.code !== 200);
            })
            .catch(error => {
                displayResult('group-result', error, true);
            });
        });
    </script>
</body>
</html>
""", 
    detection_model_path=DEFAULT_DETECTION_MODEL_PATH,
    # segclothes_model_path=DEFAULT_SEGCLOTHES_MODEL_PATH,
    segpersones_model_path=DEFAULT_SEGPERSONES_MODEL_PATH,
    inpainter_model_path=DEFAULT_INPAINTER_MODEL_PATH
)

@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({"code": 200, "message": "Service is running", "device": device})

@app.route("/api/file")
def serve_file():
    """
    Serve files from temporary and output directories
    """
    try:
        file_path = request.args.get('path')
        if not file_path:
            return jsonify(get_error(message="File path not provided")), 400
        
        # Security check - only allow serving files from allowed directories
        allowed_dirs = [TEMP_DIR, DEFAULT_OUTPUT_DIR, "/tmp"]
        is_allowed = any(os.path.commonpath([os.path.abspath(file_path), os.path.abspath(dir)]) == os.path.abspath(dir) 
                         for dir in allowed_dirs)
        
        if not is_allowed:
            return jsonify(get_error(message="Access to this file is not allowed")), 403
        
        if not os.path.exists(file_path):
            return jsonify(get_error(message="File not found")), 404
            
        return send_file(file_path)
    except Exception as e:
        return jsonify(get_error(message=f"Error serving file: {str(e)}")), 500

@app.route("/api/face/build_database", methods=["POST"])
def build_face_database():
    """
    Build face database from a folder of images
    Form data:
    - face_db_folder: path to face database folder
    """
    try:
        global face_recognition_model
        
        # Check if face recognition model is initialized
        if face_recognition_model is None:
            return jsonify(get_error(message="Face recognition model not initialized")), 500
        
        # Get parameters
        face_db_folder = request.form.get('face_db_folder')
        
        if not face_db_folder:
            return jsonify(get_error(message="face_db_folder is required")), 400
        
        if not os.path.exists(face_db_folder):
            return jsonify(get_error(message=f"Face database folder does not exist: {face_db_folder}")), 400
        
        # Build face database
        person_dirs = [d for d in os.listdir(face_db_folder) 
                      if os.path.isdir(os.path.join(face_db_folder, d))]
        
        # This will rebuild the database
        face_recognition_model.build_face_database(face_db_folder, first_run=True)
        
        # Return results
        response = {
            "code": 200,
            "message": "Face database built successfully",
            "data": {
                "face_db_built": True,
                "person_count": len(person_dirs)
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify(get_error(message=f"Error building face database: {str(e)}")), 500

@app.route("/api/face/recognize", methods=["POST"])
def recognize_faces():
    """
    Recognize faces in an image using the built face database
    Form data:
    - image: image file
    """
    temp_dir = None
    
    try:
        global face_recognition_model
        
        # Check if face recognition model is initialized
        if face_recognition_model is None:
            return jsonify(get_error(message="Face recognition model not initialized")), 500
        
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify(get_error(message="No image provided")), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify(get_error(message="No image selected")), 400
        
        # Save image to temporary file
        filename = secure_filename(image_file.filename)
        temp_dir = tempfile.mkdtemp(dir=TEMP_DIR)
        image_path = os.path.join(temp_dir, filename)
        image_file.save(image_path)
        
        # DEBUG: Print image info
        print(f"Saved uploaded image to: {image_path}")
        image = cv2.imread(image_path)
        if image is not None:
            print(f"Image loaded successfully. Dimensions: {image.shape}")
        else:
            print("ERROR: Failed to load image with cv2.imread")
            return jsonify(get_error(message="Failed to load image")), 500
        
        # Recognize faces with more verbose output
        print("Calling recognize_face method...")
        # Recognize faces（ArcFace 余弦相似度阈值）
        results, annotated_image = face_recognition_model.recognize_face(image_path, known_threshold=0.55, unknown_threshold=0.4)
        print(f"Recognition complete. Found {len(results)} faces.")

        # 保存带注释的图像
        annotated_image_path = os.path.join(temp_dir, f"annotated_{filename}")
        cv2.imwrite(annotated_image_path, annotated_image)

        # Save face images and prepare response
        recognized_faces = []
        face_images_dir = os.path.join(DEFAULT_OUTPUT_DIR, "face_recognition_results")
        os.makedirs(face_images_dir, exist_ok=True)
        
        for i, result in enumerate(results):
            # Save face image
            face_filename = f"face_{i+1}_{os.path.splitext(filename)[0]}.jpg"
            face_image_path = os.path.join(face_images_dir, face_filename)
            cv2.imwrite(face_image_path, result['face_image'])
            
            recognized_faces.append({
                "face_id": i+1,
                "identified_as": result.get('identified_as', 'Unknown'),
                "identification_confidence": float(result.get('identification_confidence', 0)),
                "face_image_path": face_image_path,
                "bbox": result['bbox'],
                "area": int(result['area'])
            })
        annotated_result_filename = f"annotated_{os.path.splitext(filename)[0]}.jpg"
        annotated_result_path = os.path.join(face_images_dir, annotated_result_filename)
        cv2.imwrite(annotated_result_path, annotated_image)
        # Return results
        response = {
            "code": 200,
            "message": "Face recognition completed successfully",
            "data": {
                "recognized_faces": recognized_faces,
                "annotated_image_path": annotated_result_path
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify(get_error(message=f"Error in face recognition: {str(e)}")), 500
    
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Failed to delete temporary directory {temp_dir}: {e}")

@app.route("/api/progress/<task_id>")
def get_progress(task_id):
    """Get progress of a task"""
    if task_id in progress_tracker:
        return jsonify(progress_tracker[task_id])
    else:
        return jsonify({"status": "not_found"}), 404

@app.route("/api/image/process", methods=["POST"])
def process_image():
    """
    Combined endpoint for person detection, clothes segmentation, and image inpainting
    Form data:
    - image: image file
    """
    try:
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify(get_error(message="No image provided")), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify(get_error(message="No image selected")), 400
        
        # Save image to temporary file
        filename = secure_filename(image_file.filename)
        temp_dir = tempfile.mkdtemp(dir=TEMP_DIR)
        # temp_dir = prepare_temp_directory()  # Create a temporary directory
        image_path = os.path.join(temp_dir, filename)
        image_file.save(image_path)
        image_p = cv2.imread(image_path)
        # Use preset model paths directly
        detection_model_path = DEFAULT_DETECTION_MODEL_PATH
        # segclothes_model_path = DEFAULT_SEGCLOTHES_MODEL_PATH
        segpersones_model_path = DEFAULT_SEGPERSONES_MODEL_PATH
        inpainter_model_path = DEFAULT_INPAINTER_MODEL_PATH
        pose_model_path = DEFAULT_POSE_MODEL_PATH

        # Initialize detection model if needed
        global detection_model
        if detection_model is None or not hasattr(detection_model, 'model_path') or detection_model.model_path != detection_model_path:
            detection_model = PersonMaskCreator(detection_model_path, 0.4)
        
        # Step 1: Detect persons
        results = detection_model.detect_persons_in_image(image_p)
        plotted_image = results[0].plot()
        plotted_save_path = os.path.join(temp_dir, f"detect_{os.path.splitext(filename)[0]}.png")
        cv2.imwrite(plotted_save_path, plotted_image)
        serialized_results = serialize_results(results)
        
        # Generate mask
        mask_filename = f"mask_{os.path.splitext(filename)[0]}.png"
        mask_output_path = os.path.join(temp_dir, mask_filename)
        mask = detection_model.generate_and_save_mask_from_results(image_p, results, mask_output_path)
        
        # Initialize segmentation model if needed
        # global segclothes_model
        # if segclothes_model is None or not hasattr(segclothes_model, 'model_path') or segclothes_model.model_path != segclothes_model_path:
        #     segclothes_model = ClothesSegmenter(segclothes_model_path)
        global segpersones_model
        if segpersones_model is None or not hasattr(segpersones_model, 'model_path') or segpersones_model.model_path != segpersones_model_path:
            segpersones_model = PersonesSegmenter(segpersones_model_path)
        
        # Step 2: Segment clothes
        # pred_seg = segclothes_model.segment_with_yolo(image_path, results)
        pred_seg, max_mask_info = segpersones_model.segment_with_yolo(image_p, results, batch_size=8)
        # save pred_seg
        seg_filename = f"seg_{os.path.splitext(filename)[0]}.png"
        seg_filepath = os.path.join(temp_dir, seg_filename)
        cv2.imwrite(seg_filepath, pred_seg)

        # extracted_img = segclothes_model.extract_segmented_area(image_path, pred_seg)
        extracted_img = segpersones_model.generate_contour_overlay_effect(
                            image_p, 
                            pred_seg, 
                            overlay_color=(128, 128, 128),  # 灰色
                            alpha=0.6  # 80% 透明度
                        )
        
        # Save extracted image
        extract_filename = f"extract_{os.path.splitext(filename)[0]}.png"
        extract_output_path = os.path.join(temp_dir, extract_filename)
        cv2.imwrite(extract_output_path, extracted_img)

        #Initialize pose model  
        global pose_model
        if pose_model is None or not hasattr(pose_model, 'model_path') or pose_model.model_path != pose_model_path:
            pose_model = ShotTypeClassifier(pose_model_path)

        # Step 3:shot type classification 
        
        result_dict,  pose_results= pose_model.classify_shot_type(
            image_p, filename, max_mask_info['max_mask_ratio'], max_mask_info['max_mask_box'])
        
        # save pose result
        pose_filename = f"pose_{os.path.splitext(filename)[0]}.png"
        pose_output_path = os.path.join(temp_dir, pose_filename)
        # Determine if we need to apply an offset (when using cropped image)
        if max_mask_info['max_mask_box'] is not None:
            x1, y1, x2, y2 = map(int, max_mask_info['max_mask_box'])
            # Add margin like in classify_shot_type
            h, w = image_p.shape[:2]
            margin = int((x2 - x1 + y2 - y1) / 10)
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            offset = (x1, y1)
        else:
            offset = (0, 0)

        pose_image = pose_model.draw_pose_result(
            image_p, pose_results, pose_output_path, offset=offset)
        
        # Initialize inpainting model if needed
        # global inpainter_model
        # if inpainter_model is None or not hasattr(inpainter_model, 'model_path') or inpainter_model.model_path != inpainter_model_path:
        #     inpainter_model = ImageInpainter(inpainter_model_path, max_size=2048)
        
        # Step 4: Inpaint image
        inpainted_image = inpainter_model.inpaint(image_path, mask_output_path)  # seg_filepath:masks, mask_output_path: boxes
        
        # Save inpainted image
        inpaint_filename = f"inpainted_{os.path.splitext(filename)[0]}.png"
        inpainted_output_path = os.path.join(temp_dir, inpaint_filename)
        cv2.imwrite(inpainted_output_path, inpainted_image)
        
        # Return results
        response = {
            "code": 200,
            "message": "All processing steps completed successfully",
            "data": {
                "original_path": image_path,
                "detection_results": serialized_results,
                # "mask_path": mask_output_path,
                "detect_path": plotted_save_path, 
                "shot_type": result_dict.get("shot_type", "Unknown"),
                "area_ratio": result_dict.get("area_ratio", 0),
                # "confidence": result_dict.get("confidence", 0),
                "extracted_path": extract_output_path,
                "pose_path": pose_output_path,
                "inpainted_path": inpainted_output_path
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify(get_error(message=f"Error in image processing: {str(e)}")), 500

@app.route("/api/embedding/build_index", methods=["POST"])
def build_embedding_index():
    """
    Build image embedding index
    Form data:
    - image_folder: path to image folder
    - index_save_path: path to save index
    - model_name (optional): embedding model name (default: resnet50)
    """
    try:
        # Get parameters
        image_folder = request.form.get('image_folder')
        index_save_path = request.form.get('index_save_path')
        model_name = request.form.get('model_name', 'resnet50')
        
        if not image_folder or not index_save_path:
            return jsonify(get_error(message="image_folder and index_save_path are required")), 400
        
        if not os.path.exists(image_folder):
            return jsonify(get_error(message=f"Image folder does not exist: {image_folder}")), 400
        
        # Build index
        vector_db, embedder, index_save_dir = build_index(image_folder, index_save_path, model_name, use_optimized=True)
        
        # Update global index path
        global last_index_path
        last_index_path = index_save_dir
        
        # Return results
        response = {
            "code": 200,
            "message": "Embedding index built successfully",
            "data": {
                "index_path": index_save_dir,
                "model_name": model_name,
                "image_count": len(vector_db.image_paths) if hasattr(vector_db, 'image_paths') else 0
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify(get_error(message=f"Error building embedding index: {str(e)}")), 500

@app.route("/api/embedding/search", methods=["POST"])
def search_similar_images():
    """
    Search for similar images
    Form data:
    - query_image: query image file
    - index_path: path to embedding index (optional, uses last built index if not provided)
    - top_k (optional): number of results (default: 5)
    """
    temp_dir = None

    try:
        # Check if query image is provided
        if 'query_image' not in request.files:
            return jsonify(get_error(message="No query image provided")), 400
        
        query_image_file = request.files['query_image']
        if query_image_file.filename == '':
            return jsonify(get_error(message="No query image selected")), 400
        
        # Get parameters
        index_path = request.form.get('index_path')
        top_k = int(request.form.get('top_k', 5))
        
        # Use last built index if not provided
        if not index_path:
            index_path = last_index_path
            
        if not index_path:
            return jsonify(get_error(message="No index path provided and no previous index built. Please build an index first.")), 400
        
        if not os.path.exists(index_path):
            return jsonify(get_error(message=f"Index path does not exist: {index_path}")), 400
        
        # Save query image to temporary file
        temp_dir = tempfile.mkdtemp(dir=TEMP_DIR)
        # temp_dir = prepare_temp_directory()  # Create a temporary directory
        query_filename = secure_filename(query_image_file.filename)
        query_image_path = os.path.join(temp_dir, query_filename)
        query_image_file.save(query_image_path)
        
        # Search for similar images
        results, merged_img, model_name = search_image(query_image_path, index_path, top_k=top_k)
        
        # Save the merged image to a specific location
        if results and merged_img:
            # Create search results directory
            search_results_dir = os.path.join(DEFAULT_OUTPUT_DIR, "search_results", model_name)
            # If directory exists, create a new directory with incremental suffix
            if os.path.exists(search_results_dir):
                i = 2
                while os.path.exists(f"{search_results_dir}_{i}"):
                    i += 1
                search_results_dir = f"{search_results_dir}_{i}"
            os.makedirs(search_results_dir, exist_ok=True)
            
            # Generate a unique filename for the combined result
            query_name = os.path.splitext(os.path.basename(query_image_path))[0]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            combined_result_filename = f"result_{query_name}_{timestamp}.jpg"
            combined_result_path = os.path.join(search_results_dir, combined_result_filename)
            
            # Save the merged image
            merged_img.save(combined_result_path)
        else:
            combined_result_path = None
        
        # Extract similar image paths from results
        similar_image_names = []
        if results:
            similar_image_names = [os.path.basename(result['image_path']) for result in results]
        
        # Return results
        response = {
            "code": 200,
            "message": "Image search completed successfully",
            "data": {
                # "query_image_path": query_image_path,
                "results": results,
                "combined_result_path": combined_result_path,
                "similar_image_names": similar_image_names
            }
        }
        # Delete the temporary directory:temp_dir

        
        return jsonify(response)
        
    except Exception as e:
        return jsonify(get_error(message=f"Error in image search: {str(e)}")), 500
    
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"Warning: Failed to delete temporary directory {temp_dir}: {e}")

@app.route("/api/embedding/group_similar", methods=["POST"])
def group_similar_images():
    """
    Group similar images from a folder
    Form data:
    - image_folder: path to image folder
    - model_name (optional): embedding model name (default: resnet50)
    - save_dir (optional): directory to save results (default: ./outputs/similar_groups)
    """
    try:
        # Get parameters
        image_folder = request.form.get('image_folder')
        model_name = request.form.get('model_name', 'resnet50')
        save_dir = request.form.get('save_dir', 'similar_groups')
        
        if not image_folder:
            return jsonify(get_error(message="image_folder is required")), 400
        
        if not os.path.exists(image_folder):
            return jsonify(get_error(message=f"Image folder does not exist: {image_folder}")), 400
        
        # Group similar images
        groups, collage_paths = find_similar_image_groups_from_folder(image_folder, model_name, images_per_group = 4, save_dir=save_dir, min_cluster_size=5)
        
        # Prepare group information
        group_info = []
        
        # Prepare detailed group information
        for i, group in enumerate(groups):
            # Get just the filenames, not full paths
            image_names = [os.path.basename(img_path) for img_path in group]
            
            group_info.append({
                "group_id": i + 1,
                "image_count": len(group),
                "image_names": image_names
            })

        
        # Return results
        response = {
            "code": 200,
            "message": "Similar image grouping completed successfully",
            "data": {
                "group_count": len(groups),
                "groups": group_info,  # Detailed group information
                "collage_paths": collage_paths  # Paths to collage images
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify(get_error(message=f"Error grouping similar images: {str(e)}")), 500

if __name__ == "__main__":
    print("Initializing models...")
    # Initialize models
    initialize_models()
    
    print("Starting Flask server on http://0.0.0.0:8198")
    # Run Flask app
    app.run(host="0.0.0.0", port=8198, debug=True)