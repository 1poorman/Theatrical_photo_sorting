# -*- coding: utf-8 -*-
"""
main.py - FastAPI API 定义

职责：
- 定义 FastAPI 应用与所有 API 路由（输入输出）
- 模型实例、默认路径与工具函数从 server 模块导入（通过 srv.xxx 访问，
  保证跨模块共享同一模型实例，重新赋值可同步到 server 模块）
- 移除自定义前端页面，使用 FastAPI 自带 /docs (Swagger UI) 交互界面

启动方式：python app/server.py
"""
import os
import sys
import datetime
import time
import tempfile
import shutil
import uuid
import cv2
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

import server as srv

from tools.logger import get_app_logger, StepTimer, log_key_action
from embedding.image_search_system_module import (
    build_index, search_image, find_similar_image_groups_from_folder
)

app_logger = get_app_logger()

app = FastAPI(
    title="Image Processing Toolkit",
    description="戏剧照片处理工具 API（人脸识别 / 人像检测分割 / 图像修复 / 图像检索聚类）",
    version="2.0.0",
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- 请求日志中间件 ----------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    client = request.client.host if request.client else "-"
    app_logger.info(
        "%s %s %s -> %s (%.2fs)",
        client, request.method, request.url.path, response.status_code, duration,
    )
    return response


# ---------- 全局异常处理 ----------

@app.exception_handler(Exception)
async def handle_exception(request: Request, exc: Exception):
    import traceback
    app_logger.error(
        "Unhandled exception on %s %s: %s",
        request.method, request.url.path, exc, exc_info=True,
    )
    traceback.print_exc()
    return JSONResponse(status_code=500, content=srv.get_error(message=str(exc)))


# ---------- 基础路由 ----------

@app.get("/", include_in_schema=False)
async def index():
    """重定向到 FastAPI 自带交互文档 /docs"""
    return RedirectResponse(url="/docs")


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return JSONResponse({"code": 200, "message": "Service is running", "device": srv.device})


@app.get("/api/file")
async def serve_file(path: str):
    """提供文件访问（临时目录 / 输出目录 / 项目根目录）"""
    try:
        if not path:
            return JSONResponse(status_code=400, content=srv.get_error(message="File path not provided"))

        # 允许目录：
        # - 临时目录与输出目录（人脸标注图、检索结果拼图等）
        # - 项目根目录（检索索引指向的原始图片文件夹可能位于项目内，
        #   如 data/ncpa_test，需允许其缩略图在前端展示）
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        allowed_dirs = [srv.TEMP_DIR, srv.DEFAULT_OUTPUT_DIR, "/tmp", project_root,
                        os.path.join(project_root, "data")]
        is_allowed = any(
            os.path.commonpath([os.path.abspath(path), os.path.abspath(d)]) == os.path.abspath(d)
            for d in allowed_dirs
        )
        if not is_allowed:
            return JSONResponse(status_code=403, content=srv.get_error(message="Access to this file is not allowed"))
        if not os.path.exists(path):
            return JSONResponse(status_code=404, content=srv.get_error(message="File not found"))
        return FileResponse(path)
    except Exception as e:
        return JSONResponse(status_code=500, content=srv.get_error(message=f"Error serving file: {str(e)}"))


@app.get("/api/progress/{task_id}")
async def get_progress(task_id: str):
    """查询任务进度"""
    if task_id in srv.progress_tracker:
        return JSONResponse(srv.progress_tracker[task_id])
    else:
        return JSONResponse(status_code=404, content={"status": "not_found"})


# ---------- 人脸识别 ----------

@app.post("/api/face/build_database")
async def build_face_database(face_db_folder: str = Form(...)):
    """构建人脸库
    Form 参数:
    - face_db_folder: 人脸库文件夹路径（子目录按人物命名）
    """
    try:
        if srv.face_recognition_model is None:
            return JSONResponse(status_code=500, content=srv.get_error(message="Face recognition model not initialized"))
        if not face_db_folder:
            return JSONResponse(status_code=400, content=srv.get_error(message="face_db_folder is required"))
        if not os.path.exists(face_db_folder):
            return JSONResponse(status_code=400, content=srv.get_error(message=f"Face database folder does not exist: {face_db_folder}"))

        person_dirs = [d for d in os.listdir(face_db_folder)
                       if os.path.isdir(os.path.join(face_db_folder, d))]
        srv.face_recognition_model.build_face_database(face_db_folder, first_run=True)

        return JSONResponse({
            "code": 200,
            "message": "Face database built successfully",
            "data": {"face_db_built": True, "person_count": len(person_dirs)}
        })
    except Exception as e:
        return JSONResponse(status_code=500, content=srv.get_error(message=f"Error building face database: {str(e)}"))


@app.post("/api/face/recognize")
async def recognize_faces(image: UploadFile = File(...)):
    """识别图片中的人脸
    Form 参数:
    - image: 待识别的图片文件
    """
    temp_dir = None
    try:
        if srv.face_recognition_model is None:
            return JSONResponse(status_code=500, content=srv.get_error(message="Face recognition model not initialized"))
        if image is None or image.filename == '':
            return JSONResponse(status_code=400, content=srv.get_error(message="No image provided"))

        from werkzeug.utils import secure_filename
        filename = secure_filename(image.filename)
        temp_dir = tempfile.mkdtemp(dir=srv.TEMP_DIR)
        image_path = os.path.join(temp_dir, filename)
        with open(image_path, "wb") as f:
            f.write(await image.read())

        img_check = cv2.imread(image_path)
        if img_check is None:
            print("ERROR: Failed to load image with cv2.imread")
            return JSONResponse(status_code=500, content=srv.get_error(message="Failed to load image"))

        # 识别人脸（ArcFace 余弦相似度阈值）
        results, annotated_image = srv.face_recognition_model.recognize_face(
            image_path, known_threshold=0.55, unknown_threshold=0.4)
        print(f"Recognition complete. Found {len(results)} faces.")

        # 保存人脸图片与标注图
        face_images_dir = os.path.join(srv.DEFAULT_OUTPUT_DIR, "face_recognition_results")
        os.makedirs(face_images_dir, exist_ok=True)

        recognized_faces = []
        for i, result in enumerate(results):
            face_filename = f"face_{i+1}_{os.path.splitext(filename)[0]}.jpg"
            face_image_path = os.path.join(face_images_dir, face_filename)
            cv2.imwrite(face_image_path, result['face_image'])
            recognized_faces.append({
                "face_id": i + 1,
                "identified_as": result.get('identified_as', 'Unknown'),
                "identification_confidence": float(result.get('identification_confidence', 0)),
                "face_image_path": face_image_path,
                "bbox": result['bbox'],
                "area": int(result['area'])
            })

        annotated_result_path = os.path.join(face_images_dir, f"annotated_{os.path.splitext(filename)[0]}.jpg")
        cv2.imwrite(annotated_result_path, annotated_image)

        return JSONResponse({
            "code": 200,
            "message": "Face recognition completed successfully",
            "data": {"recognized_faces": recognized_faces, "annotated_image_path": annotated_result_path}
        })
    except Exception as e:
        return JSONResponse(status_code=500, content=srv.get_error(message=f"Error in face recognition: {str(e)}"))
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Failed to delete temporary directory {temp_dir}: {e}")


# ---------- 人像检测 / 分割 / 修复 ----------

@app.post("/api/image/process")
async def process_image(image: UploadFile = File(...)):
    """组合处理：人像检测 + 服装分割 + 镜头分类 + 图像修复
    Form 参数:
    - image: 待处理图片文件
    """
    temp_dir = None
    timer = StepTimer(app_logger, task_id=uuid.uuid4().hex[:8])
    try:
        if image is None or image.filename == '':
            return JSONResponse(status_code=400, content=srv.get_error(message="No image provided"))

        from werkzeug.utils import secure_filename
        filename = secure_filename(image.filename)
        with timer.step("prepare"):
            temp_dir = tempfile.mkdtemp(dir=srv.TEMP_DIR)
            image_path = os.path.join(temp_dir, filename)
            with open(image_path, "wb") as f:
                f.write(await image.read())
            image_p = cv2.imread(image_path)
            if image_p is None:
                raise ValueError(f"Failed to load image: {image_path}")
            app_logger.info("[%s] input image: %s (%dx%d)", timer.task_id, filename, image_p.shape[1], image_p.shape[0])

        # 使用预设模型路径
        detection_model_path = srv.DEFAULT_DETECTION_MODEL_PATH
        segpersones_model_path = srv.DEFAULT_SEGPERSONES_MODEL_PATH
        inpainter_model_path = srv.DEFAULT_INPAINTER_MODEL_PATH
        pose_model_path = srv.DEFAULT_POSE_MODEL_PATH

        # Step 1: 人像检测
        with timer.step("detect_persons"):
            if srv.detection_model is None or not hasattr(srv.detection_model, 'model_path') or srv.detection_model.model_path != detection_model_path:
                app_logger.warning("[%s] detection model (re)loading: %s", timer.task_id, detection_model_path)
                srv.detection_model = srv.PersonMaskCreator(detection_model_path, 0.4)

            results = srv.detection_model.detect_persons_in_image(image_p)
            plotted_image = results[0].plot()
            plotted_save_path = os.path.join(temp_dir, f"detect_{os.path.splitext(filename)[0]}.png")
            cv2.imwrite(plotted_save_path, plotted_image)
            serialized_results = srv.serialize_results(results)
            n_persons = len(results[0].boxes) if results and results[0].boxes is not None else 0
            app_logger.info("[%s] detection: %d person(s)", timer.task_id, n_persons)

            mask_output_path = os.path.join(temp_dir, f"mask_{os.path.splitext(filename)[0]}.png")
            srv.detection_model.generate_and_save_mask_from_results(image_p, results, mask_output_path)

        # Step 2: 服装分割
        with timer.step("segment"):
            if srv.segpersones_model is None or not hasattr(srv.segpersones_model, 'model_path') or srv.segpersones_model.model_path != segpersones_model_path:
                app_logger.warning("[%s] segmentation model (re)loading: %s", timer.task_id, segpersones_model_path)
                srv.segpersones_model = srv.PersonesSegmenter(segpersones_model_path)

            pred_seg, max_mask_info = srv.segpersones_model.segment_with_yolo(image_p, results, batch_size=8)
            seg_filepath = os.path.join(temp_dir, f"seg_{os.path.splitext(filename)[0]}.png")
            cv2.imwrite(seg_filepath, pred_seg)

            extracted_img = srv.segpersones_model.generate_contour_overlay_effect(
                image_p, pred_seg, overlay_color=(128, 128, 128), alpha=0.6)
            extract_output_path = os.path.join(temp_dir, f"extract_{os.path.splitext(filename)[0]}.png")
            cv2.imwrite(extract_output_path, extracted_img)
            app_logger.info("[%s] segmentation: max_mask_ratio=%.3f", timer.task_id, max_mask_info.get('max_mask_ratio', 0.0))

        # Step 3: 镜头分类
        with timer.step("shot_classify"):
            if srv.pose_model is None or not hasattr(srv.pose_model, 'model_path') or srv.pose_model.model_path != pose_model_path:
                app_logger.warning("[%s] pose model (re)loading: %s", timer.task_id, pose_model_path)
                srv.pose_model = srv.ShotTypeClassifier(pose_model_path)

            result_dict, pose_results = srv.pose_model.classify_shot_type(
                image_p, filename, max_mask_info['max_mask_ratio'], max_mask_info['max_mask_box'])

            pose_output_path = os.path.join(temp_dir, f"pose_{os.path.splitext(filename)[0]}.png")
            # 姿态估计在整图上进行，坐标无需偏移；绘制分类时选定的主要人物
            srv.pose_model.draw_pose_result(
                image_p, pose_results, pose_output_path,
                offset=(0, 0), person_idx=result_dict.get("main_person_idx"))
            app_logger.info("[%s] shot classification: %s (conf=%.2f)",
                            timer.task_id, result_dict.get("shot_type"), result_dict.get("confidence", 0))

        # Step 4: 图像修复
        with timer.step("inpaint"):
            if srv.inpainter_model is None:
                app_logger.warning("[%s] inpainter model (re)loading: %s", timer.task_id, srv.DEFAULT_INPAINTER_MODEL_PATH)
                srv.inpainter_model = srv.ImageInpainter(srv.DEFAULT_INPAINTER_MODEL_PATH, max_size=1024)
            inpainted_image = srv.inpainter_model.inpaint(image_path, mask_output_path)
            inpainted_output_path = os.path.join(temp_dir, f"inpainted_{os.path.splitext(filename)[0]}.png")
            cv2.imwrite(inpainted_output_path, inpainted_image)

        timer.log_summary(extra=f"file={filename}")
        return JSONResponse({
            "code": 200,
            "message": "All processing steps completed successfully",
            "data": {
                "original_path": image_path,
                "detection_results": serialized_results,
                "detect_path": plotted_save_path,
                "shot_type": result_dict.get("shot_type", "Unknown"),
                "area_ratio": result_dict.get("area_ratio", 0),
                "extracted_path": extract_output_path,
                "pose_path": pose_output_path,
                "inpainted_path": inpainted_output_path
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        timer.log_summary(extra=f"file={getattr(image, 'filename', 'unknown')} error={str(e)}")
        return JSONResponse(status_code=500, content=srv.get_error(message=f"Error in image processing: {str(e)}"))


# ---------- 图像嵌入检索 ----------

@app.post("/api/embedding/build_index")
async def build_embedding_index(image_folder: str = Form(...),
                                index_save_path: str = Form(...),
                                model_name: str = Form("resnet50")):
    """构建图像 embedding 索引
    Form 参数:
    - image_folder: 图片文件夹路径
    - index_save_path: 索引保存路径
    - model_name: 嵌入模型名称 (默认 resnet50)
    """
    try:
        if not image_folder or not index_save_path:
            return JSONResponse(status_code=400, content=srv.get_error(message="image_folder and index_save_path are required"))
        if not os.path.exists(image_folder):
            return JSONResponse(status_code=400, content=srv.get_error(message=f"Image folder does not exist: {image_folder}"))

        vector_db, embedder, index_save_dir = build_index(
            image_folder, index_save_path, model_name, use_optimized=True)
        srv.last_index_path = index_save_dir

        return JSONResponse({
            "code": 200,
            "message": "Embedding index built successfully",
            "data": {
                "index_path": index_save_dir,
                "model_name": model_name,
                "image_count": len(vector_db.image_paths) if hasattr(vector_db, 'image_paths') else 0
            }
        })
    except Exception as e:
        return JSONResponse(status_code=500, content=srv.get_error(message=f"Error building embedding index: {str(e)}"))


@app.post("/api/embedding/search")
async def search_similar_images(query_image: UploadFile = File(...),
                                index_path: str = Form(None),
                                top_k: int = Form(5)):
    """相似图片检索
    Form 参数:
    - query_image: 查询图片文件
    - index_path: 索引路径（可选，默认使用最近构建的索引）
    - top_k: 返回结果数量 (默认 5)
    """
    temp_dir = None
    try:
        if query_image is None or query_image.filename == '':
            return JSONResponse(status_code=400, content=srv.get_error(message="No query image provided"))

        if not index_path:
            index_path = srv.last_index_path
        if not index_path:
            return JSONResponse(status_code=400, content=srv.get_error(
                message="No index path provided and no previous index built. Please build an index first."))
        if not os.path.exists(index_path):
            return JSONResponse(status_code=400, content=srv.get_error(message=f"Index path does not exist: {index_path}"))

        from werkzeug.utils import secure_filename
        temp_dir = tempfile.mkdtemp(dir=srv.TEMP_DIR)
        query_filename = secure_filename(query_image.filename)
        query_image_path = os.path.join(temp_dir, query_filename)
        with open(query_image_path, "wb") as f:
            f.write(await query_image.read())

        results, merged_img, model_name = search_image(query_image_path, index_path, top_k=top_k)

        combined_result_path = None
        if results and merged_img:
            search_results_dir = os.path.join(srv.DEFAULT_OUTPUT_DIR, "search_results", model_name)
            if os.path.exists(search_results_dir):
                i = 2
                while os.path.exists(f"{search_results_dir}_{i}"):
                    i += 1
                search_results_dir = f"{search_results_dir}_{i}"
            os.makedirs(search_results_dir, exist_ok=True)

            query_name = os.path.splitext(os.path.basename(query_image_path))[0]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            combined_result_path = os.path.join(
                search_results_dir, f"result_{query_name}_{timestamp}.jpg")
            merged_img.save(combined_result_path)

        similar_image_names = [os.path.basename(result['image_path']) for result in results] if results else []

        return JSONResponse({
            "code": 200,
            "message": "Image search completed successfully",
            "data": {
                "results": results,
                "combined_result_path": combined_result_path,
                "similar_image_names": similar_image_names
            }
        })
    except Exception as e:
        return JSONResponse(status_code=500, content=srv.get_error(message=f"Error in image search: {str(e)}"))
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Failed to delete temporary directory {temp_dir}: {e}")


@app.post("/api/embedding/group_similar")
async def group_similar_images(image_folder: str = Form(...),
                               model_name: str = Form("resnet50"),
                               save_dir: str = Form("similar_groups"),
                               n_clusters: Optional[int] = Form(None),
                               min_cluster_size: int = Form(2),
                               images_per_group: int = Form(4)):
    """相似图片聚类分组
    Form 参数:
    - image_folder: 图片文件夹路径
    - model_name: 嵌入模型名称 (默认 resnet50)
    - save_dir: 结果保存目录 (默认 ./outputs/similar_groups)
    - n_clusters: 聚类数量 (可选，留空则通过轮廓系数自动选择最优值)
    - min_cluster_size: 每个聚类最少图片数，小于该值的聚类被过滤 (默认 2)
    - images_per_group: 每组拼图中展示的代表图片数 (默认 4)
    """
    try:
        if not image_folder:
            return JSONResponse(status_code=400, content=srv.get_error(message="image_folder is required"))
        if not os.path.exists(image_folder):
            return JSONResponse(status_code=400, content=srv.get_error(message=f"Image folder does not exist: {image_folder}"))

        groups, collage_paths, per_group_collages = find_similar_image_groups_from_folder(
            image_folder, model_name,
            images_per_group=images_per_group,
            save_dir=save_dir,
            min_cluster_size=min_cluster_size,
            n_clusters=n_clusters)

        group_info = []
        for i, group in enumerate(groups):
            image_names = [os.path.basename(img_path) for img_path in group]
            group_collages = per_group_collages[i] if i < len(per_group_collages) else []
            group_info.append({
                "group_id": i + 1,
                "image_count": len(group),
                "image_names": image_names,
                "collage_paths": group_collages,
                "collage_count": len(group_collages)
            })

        return JSONResponse({
            "code": 200,
            "message": "Similar image grouping completed successfully",
            "data": {
                "group_count": len(groups),
                "groups": group_info,
                "collage_paths": collage_paths
            }
        })
    except Exception as e:
        return JSONResponse(status_code=500, content=srv.get_error(message=f"Error grouping similar images: {str(e)}"))


# ---------- 启动事件 ----------

@app.on_event("startup")
async def startup_event():
    app_logger.info("Server startup: initializing models...")
    ok = srv.initialize_models()
    app_logger.info("Model initialization %s", "succeeded" if ok else "failed (some models may be missing)")
