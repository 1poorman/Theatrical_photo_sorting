# myserver.py
import os
import cv2
import torch
import sys
sys.path.append(".")
from flask import Flask, request, jsonify, render_template_string, send_file, url_for
from werkzeug.utils import secure_filename
import tempfile
import shutil
import json

from detection.PersonMaskCreator import PersonMaskCreator
from inpaint.inpaint_lama import ImageInpainter
from seg_clothes.ClothesSegment import ClothesSegmenter
from embedding.image_search_system_module import (
    build_index, search_image, find_similar_image_groups_from_folder
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Max 100MB upload

# Global model instances
detection_model = None
segclothes_model = None
inpainter_model = None

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Default paths - these should be configured based on your environment
DEFAULT_DETECTION_MODEL_PATH = "/home/huachenghao/codes/Sorting_theatrical-photo/detection/yolo11l.pt"
DEFAULT_SEGCLOTHES_MODEL_PATH = "/home/huachenghao/codes/clothes/models--mattmdjaga--segformer_b2_clothes/snapshots/584abc1e1d260e23c0fc627c5217a09b2b461046"
DEFAULT_INPAINTER_MODEL_PATH = "/home/huachenghao/codes/cv_fft_inpainting_lama"
DEFAULT_OUTPUT_DIR = "./outputs"

# Create directories if they don't exist
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
TEMP_DIR = os.path.join(DEFAULT_OUTPUT_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

def initialize_models(detection_model_path=DEFAULT_DETECTION_MODEL_PATH,
                     segclothes_model_path=DEFAULT_SEGCLOTHES_MODEL_PATH,
                     inpainter_model_path=DEFAULT_INPAINTER_MODEL_PATH):
    """
    Initialize all models
    """
    global detection_model, segclothes_model, inpainter_model
    
    try:
        if detection_model_path and os.path.exists(detection_model_path):
            detection_model = PersonMaskCreator(detection_model_path)
            print(f"Detection model loaded from {detection_model_path}")
        else:
            print(f"Detection model not found at {detection_model_path}")
        
        if segclothes_model_path and os.path.exists(segclothes_model_path):
            segclothes_model = ClothesSegmenter(segclothes_model_path)
            print(f"Segmentation model loaded from {segclothes_model_path}")
        else:
            print(f"Segmentation model not found at {segclothes_model_path}")
            
        if inpainter_model_path and os.path.exists(inpainter_model_path):
            inpainter_model = ImageInpainter(inpainter_model_path)
            print(f"Inpainting model loaded from {inpainter_model_path}")
        else:
            print(f"Inpainting model not found at {inpainter_model_path}")
            
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
        
        <!-- Person Detection -->
        <div class="section">
            <h2>👤 Person Detection</h2>
            <form id="detect-form" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="detect-image">Upload Image:</label>
                    <input type="file" id="detect-image" name="image" accept="image/*" required>
                </div>
                <div class="form-group">
                    <label for="detection-model-path">Detection Model Path (optional):</label>
                    <input type="text" id="detection-model-path" name="detection_model_path" value="{{ detection_model_path }}">
                </div>
                <button type="submit">Detect Persons</button>
            </form>
            <div id="detect-result" class="result"></div>
        </div>
        
        <!-- Clothes Segmentation -->
        <div class="section">
            <h2>👕 Clothes Segmentation</h2>
            <form id="segment-form" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="segment-image">Upload Image:</label>
                    <input type="file" id="segment-image" name="image" accept="image/*" required>
                </div>
                <div class="form-group">
                    <label for="detection-results">Detection Results (JSON):</label>
                    <textarea id="detection-results" name="detection_results" rows="4" placeholder='Paste detection results JSON here'></textarea>
                </div>
                <div class="form-group">
                    <label for="segclothes-model-path">Segmentation Model Path (optional):</label>
                    <input type="text" id="segclothes-model-path" name="segclothes_model_path" value="{{ segclothes_model_path }}">
                </div>
                <button type="submit">Segment Clothes</button>
            </form>
            <div id="segment-result" class="result"></div>
        </div>
        
        <!-- Image Inpainting -->
        <div class="section">
            <h2>🎨 Image Inpainting</h2>
            <form id="inpaint-form" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="inpaint-image">Original Image:</label>
                    <input type="file" id="inpaint-image" name="image" accept="image/*" required>
                </div>
                <div class="form-group">
                    <label for="inpaint-mask">Mask Image:</label>
                    <input type="file" id="inpaint-mask" name="mask" accept="image/*" required>
                </div>
                <div class="form-group">
                    <label for="inpainter-model-path">Inpainting Model Path (optional):</label>
                    <input type="text" id="inpainter-model-path" name="inpainter_model_path" value="{{ inpainter_model_path }}">
                </div>
                <button type="submit">Inpaint Image</button>
            </form>
            <div id="inpaint-result" class="result"></div>
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
                    <label for="model-name">Model Name:</label>
                    <select id="model-name" name="model_name">
                        <option value="dinov2_small">DINOv2 Small</option>
                        <option value="dinov2_base">DINOv2 Base</option>
                        <option value="dinov2_large">DINOv2 Large</option>
                        <option value="resnet50">ResNet50</option>
                        <option value="resnet101">ResNet101</option>
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
                    <label for="query-image">Query Image:</label>
                    <input type="file" id="query-image" name="query_image" accept="image/*" required>
                </div>
                <div class="form-group">
                    <label for="search-index-path">Index Path:</label>
                    <input type="text" id="search-index-path" name="index_path" placeholder="/path/to/index" required>
                </div>
                <div class="form-group">
                    <label for="top-k">Number of Results:</label>
                    <input type="number" id="top-k" name="top_k" value="5" min="1" max="20">
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
                        <option value="dinov2_small">DINOv2 Small</option>
                        <option value="dinov2_base">DINOv2 Base</option>
                        <option value="dinov2_large">DINOv2 Large</option>
                        <option value="resnet50">ResNet50</option>
                        <option value="resnet101">ResNet101</option>
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
                    if (data.data && data.data.mask_path) {
                        element.innerHTML = `
                            <strong>Success!</strong>
                            <p>Detection results: <pre>${JSON.stringify(data.data.detection_results, null, 2)}</pre></p>
                            <div class="image-container">
                                <div class="image-item">
                                    <p>Original Image</p>
                                    <img src="/api/file?path=${encodeURIComponent(document.getElementById('detect-image').files[0].name)}" alt="Original">
                                </div>
                                <div class="image-item">
                                    <p>Mask Image</p>
                                    <img src="/api/file?path=${encodeURIComponent(data.data.mask_path)}" alt="Mask">
                                </div>
                            </div>
                        `;
                    } else if (data.data && data.data.segmentation_result) {
                        element.innerHTML = `
                            <strong>Success!</strong>
                            <div class="image-container">
                                <div class="image-item">
                                    <p>Original Image</p>
                                    <img src="/api/file?path=${encodeURIComponent(document.getElementById('segment-image').files[0].name)}" alt="Original">
                                </div>
                                <div class="image-item">
                                    <p>Segmented Image</p>
                                    <img src="/api/file?path=${encodeURIComponent(data.data.segmentation_result)}" alt="Segmented">
                                </div>
                            </div>
                        `;
                    } else if (data.data && data.data.inpainted_image) {
                        element.innerHTML = `
                            <strong>Success!</strong>
                            <div class="image-container">
                                <div class="image-item">
                                    <p>Original Image</p>
                                    <img src="/api/file?path=${encodeURIComponent(document.getElementById('inpaint-image').files[0].name)}" alt="Original">
                                </div>
                                <div class="image-item">
                                    <p>Inpainted Image</p>
                                    <img src="/api/file?path=${encodeURIComponent(data.data.inpainted_image)}" alt="Inpainted">
                                </div>
                            </div>
                        `;
                    } else if (data.data && data.data.results) {
                        let resultsHtml = '<strong>Success! Search Results:</strong>';
                        if (data.data.results.length > 0) {
                            resultsHtml += '<div class="image-container">';
                            resultsHtml += `
                                <div class="image-item">
                                    <p>Query Image</p>
                                    <img src="/api/file?path=${encodeURIComponent(document.getElementById('search-query-image').files[0].name)}" alt="Query">
                                </div>
                            `;
                            
                            data.data.results.forEach((result, index) => {
                                resultsHtml += `
                                    <div class="image-item">
                                        <p>Result #${index + 1} (Similarity: ${result.similarity.toFixed(3)})</p>
                                        <img src="/api/file?path=${encodeURIComponent(result.image_path)}" alt="Result ${index + 1}">
                                    </div>
                                `;
                            });
                            resultsHtml += '</div>';
                        } else {
                            resultsHtml += '<p>No similar images found.</p>';
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
            fetch('/api/health')
                .then(response => response.json())
                .then(data => {
                    displayResult('health-result', data);
                })
                .catch(error => {
                    displayResult('health-result', error, true);
                });
        }

        // Person detection form submission
        document.getElementById('detect-form').addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            
            fetch('/api/person/detect', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                displayResult('detect-result', data, data.code !== 200);
            })
            .catch(error => {
                displayResult('detect-result', error, true);
            });
        });

        // Clothes segmentation form submission
        document.getElementById('segment-form').addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            
            fetch('/api/clothes/segment', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                displayResult('segment-result', data, data.code !== 200);
            })
            .catch(error => {
                displayResult('segment-result', error, true);
            });
        });

        // Image inpainting form submission
        document.getElementById('inpaint-form').addEventListener('submit', function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            
            fetch('/api/image/inpaint', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                displayResult('inpaint-result', data, data.code !== 200);
            })
            .catch(error => {
                displayResult('inpaint-result', error, true);
            });
        });

        // Build index form submission
        document.getElementById('index-form').addEventListener('submit', function(e) {
            e.preventDefault();
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
    segclothes_model_path=DEFAULT_SEGCLOTHES_MODEL_PATH,
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
        allowed_dirs = [TEMP_DIR, DEFAULT_OUTPUT_DIR]
        is_allowed = any(os.path.commonpath([os.path.abspath(file_path), os.path.abspath(dir)]) == os.path.abspath(dir) 
                         for dir in allowed_dirs)
        
        if not is_allowed:
            return jsonify(get_error(message="Access to this file is not allowed")), 403
        
        if not os.path.exists(file_path):
            return jsonify(get_error(message="File not found")), 404
            
        return send_file(file_path)
    except Exception as e:
        return jsonify(get_error(message=f"Error serving file: {str(e)}")), 500

@app.route("/api/person/detect", methods=["POST"])
def detect_persons():
    """
    Detect persons in an image
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
        image_path = os.path.join(temp_dir, filename)
        image_file.save(image_path)
        
        # Use preset model path directly
        detection_model_path = DEFAULT_DETECTION_MODEL_PATH
        
        # Initialize detection model if needed
        global detection_model
        if detection_model is None or not hasattr(detection_model, 'model_path') or detection_model.model_path != detection_model_path:
            detection_model = PersonMaskCreator(detection_model_path)
        
        # Detect persons
        results = detection_model.detect_persons_in_image(image_path)
        
        # Serialize results to make them JSON serializable
        serialized_results = serialize_results(results)
        
        # Generate mask
        mask_filename = f"mask_{os.path.splitext(filename)[0]}.png"
        mask_output_path = os.path.join(temp_dir, mask_filename)
        mask = detection_model.generate_and_save_mask_from_results(image_path, results, mask_output_path)
        
        # Return results
        response = {
            "code": 200,
            "message": "Person detection completed successfully",
            "data": {
                "detection_results": serialized_results,
                "mask_path": mask_output_path
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify(get_error(message=f"Error in person detection: {str(e)}")), 500

@app.route("/api/clothes/segment", methods=["POST"])
def segment_clothes():
    """
    Segment clothes in detected persons
    Form data:
    - image: image file
    - detection_results: JSON string of detection results
    """
    try:
        # Check if image and detection results are provided
        if 'image' not in request.files:
            return jsonify(get_error(message="No image provided")), 400
        
        if 'detection_results' not in request.form:
            return jsonify(get_error(message="No detection results provided")), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify(get_error(message="No image selected")), 400
        
        # Parse detection results
        detection_results = json.loads(request.form['detection_results'])
        
        # Save image to temporary file
        filename = secure_filename(image_file.filename)
        temp_dir = tempfile.mkdtemp(dir=TEMP_DIR)
        image_path = os.path.join(temp_dir, filename)
        image_file.save(image_path)
        
        # Use preset model path directly
        segclothes_model_path = DEFAULT_SEGCLOTHES_MODEL_PATH
        
        # Initialize segmentation model if needed
        global segclothes_model
        if segclothes_model is None or not hasattr(segclothes_model, 'model_path') or segclothes_model.model_path != segclothes_model_path:
            segclothes_model = ClothesSegmenter(segclothes_model_path)
        
        # Segment clothes
        pred_seg = segclothes_model.segment_with_yolo(image_path, detection_results)
        extracted_img = segclothes_model.extract_segmented_area(image_path, pred_seg)
        
        # Save extracted image
        extract_filename = f"extract_{os.path.splitext(filename)[0]}.png"
        extract_output_path = os.path.join(temp_dir, extract_filename)
        cv2.imwrite(extract_output_path, extracted_img)
        
        # Return results
        response = {
            "code": 200,
            "message": "Clothes segmentation completed successfully",
            "data": {
                "segmentation_result": extract_output_path
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify(get_error(message=f"Error in clothes segmentation: {str(e)}")), 500

@app.route("/api/image/inpaint", methods=["POST"])
def inpaint_image():
    """
    Inpaint image using mask
    Form data:
    - image: image file
    - mask: mask image file
    """
    try:
        # Check if image and mask are provided
        if 'image' not in request.files:
            return jsonify(get_error(message="No image provided")), 400
        
        if 'mask' not in request.files:
            return jsonify(get_error(message="No mask provided")), 400
        
        image_file = request.files['image']
        mask_file = request.files['mask']
        
        if image_file.filename == '' or mask_file.filename == '':
            return jsonify(get_error(message="No image or mask selected")), 400
        
        # Save image and mask to temporary files
        temp_dir = tempfile.mkdtemp(dir=TEMP_DIR)
        image_filename = secure_filename(image_file.filename)
        mask_filename = secure_filename(mask_file.filename)
        
        image_path = os.path.join(temp_dir, image_filename)
        mask_path = os.path.join(temp_dir, mask_filename)
        
        image_file.save(image_path)
        mask_file.save(mask_path)
        
        # Use preset model path directly
        inpainter_model_path = DEFAULT_INPAINTER_MODEL_PATH
        
        # Initialize inpainting model if needed
        global inpainter_model
        if inpainter_model is None or not hasattr(inpainter_model, 'model_path') or inpainter_model.model_path != inpainter_model_path:
            inpainter_model = ImageInpainter(inpainter_model_path)
        
        # Inpaint image
        inpainted_image = inpainter_model.inpaint(image_path, mask_path)
        
        # Save inpainted image
        inpaint_filename = f"inpainted_{os.path.splitext(image_filename)[0]}.png"
        inpainted_output_path = os.path.join(temp_dir, inpaint_filename)
        cv2.imwrite(inpainted_output_path, inpainted_image)
        
        # Return results
        response = {
            "code": 200,
            "message": "Image inpainting completed successfully",
            "data": {
                "inpainted_image": inpainted_output_path
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify(get_error(message=f"Error in image inpainting: {str(e)}")), 500

@app.route("/api/embedding/build_index", methods=["POST"])
def build_embedding_index():
    """
    Build image embedding index
    Form data:
    - image_folder: path to image folder
    - index_save_path: path to save index
    - model_name (optional): embedding model name (default: dinov2_small)
    """
    try:
        # Get parameters
        image_folder = request.form.get('image_folder')
        index_save_path = request.form.get('index_save_path')
        model_name = request.form.get('model_name', 'dinov2_small')
        
        if not image_folder or not index_save_path:
            return jsonify(get_error(message="image_folder and index_save_path are required")), 400
        
        if not os.path.exists(image_folder):
            return jsonify(get_error(message=f"Image folder does not exist: {image_folder}")), 400
        
        # Build index
        vector_db, embedder = build_index(image_folder, index_save_path, model_name)
        
        # Return results
        response = {
            "code": 200,
            "message": "Embedding index built successfully",
            "data": {
                "index_path": index_save_path,
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
    - index_path: path to embedding index
    - top_k (optional): number of results (default: 5)
    """
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
        
        if not index_path:
            return jsonify(get_error(message="index_path is required")), 400
        
        if not os.path.exists(index_path):
            return jsonify(get_error(message=f"Index path does not exist: {index_path}")), 400
        
        # Save query image to temporary file
        temp_dir = tempfile.mkdtemp(dir=TEMP_DIR)
        query_filename = secure_filename(query_image_file.filename)
        query_image_path = os.path.join(temp_dir, query_filename)
        query_image_file.save(query_image_path)
        
        # Search for similar images
        results = search_image(query_image_path, index_path, top_k=top_k, show_results=False, save_results=False)
        
        # Return results
        response = {
            "code": 200,
            "message": "Image search completed successfully",
            "data": {
                "results": results
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify(get_error(message=f"Error in image search: {str(e)}")), 500
    finally:
        # Note: In production, you might want to implement a cleaner for temp files
        pass

@app.route("/api/embedding/group_similar", methods=["POST"])
def group_similar_images():
    """
    Group similar images from a folder
    Form data:
    - image_folder: path to image folder
    - model_name (optional): embedding model name (default: dinov2_small)
    - save_dir (optional): directory to save results (default: similar_groups)
    """
    try:
        # Get parameters
        image_folder = request.form.get('image_folder')
        model_name = request.form.get('model_name', 'dinov2_small')
        save_dir = request.form.get('save_dir', 'similar_groups')
        
        if not image_folder:
            return jsonify(get_error(message="image_folder is required")), 400
        
        if not os.path.exists(image_folder):
            return jsonify(get_error(message=f"Image folder does not exist: {image_folder}")), 400
        
        # Group similar images
        groups, collage_paths = find_similar_image_groups_from_folder(image_folder, model_name, save_dir=save_dir)
        
        # Return results
        response = {
            "code": 200,
            "message": "Similar image grouping completed successfully",
            "data": {
                "group_count": len(groups),
                "groups": groups
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify(get_error(message=f"Error grouping similar images: {str(e)}")), 500

if __name__ == "__main__":
    print("Initializing models...")
    # Initialize models
    initialize_models()
    
    print("Starting Flask server on http://0.0.0.0:8089")
    # Run Flask app
    app.run(host="0.0.0.0", port=8089, debug=True)