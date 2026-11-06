# 在线向量库使用与迁移

Docker 默认使用 Qdrant 1.20.0 保存图像和人脸向量。本地直接运行保留 FAISS / ES 默认配置，可设置 `TRANS_IMAGE_VECTOR_BACKEND=qdrant`、`TRANS_FACE_VECTOR_BACKEND=qdrant`、`TRANS_QDRANT_URL=http://地址:6333` 切换。远端认证支持 `TRANS_QDRANT_API_KEY`；Compose 自带服务仅开放内部网络。方案与实测见 [升级方案](qdrant_upgrade_plan.md)、[对比报告](reports/qdrant_20261102/README.md)。

## 数据与检索

图像索引的 `<index_save_path>/<model_name>/vector_store.json` 保存集合名称、模型及预处理签名；向量和项目/剧目/场景信息保存在数据库。原图仍使用现有路径，所以迁移数据库时也要保持原图路径可访问。连接清单不包含完整向量，不会随图库线性增长；构建向量按 32 张一批提取和写入，路径列表仍随图片数量增长。

应用直接调用 Qdrant 在线查询，不将全库拉回 FAISS。旧 FAISS 索引及离线分组仍按现有实现运行。图像和人脸默认精确余弦检索；图像请求可传 `exact=false` 启用近似查询。HNSW 默认 m=16、ef_construct=100、查询 ef=128；小集合可能走精确扫描。已测随机高维数据存在明显近似召回损失，因此没有将 ANN 设为默认。人脸阈值和人物去重规则保持原语义。

图像索引按模型和预处理签名校验，人脸集合按 ArcFace 权重签名隔离；更换模型后需重建。服务不可用时返回错误，不会静默使用旧索引结果。

## HTTP 操作

下例使用容器内路径，图片目录须已存在。HTTP 默认端口 8199。

```bash
curl -X POST http://localhost:8199/api/embedding/build_index \
  -F image_folder=/app/data/照片目录 \
  -F index_save_path=/app/outputs/my_index \
  -F model_name=siglip2_base \
  -F 'metadata_json={"project_id":"project-a","play_id":"play-a"}'

curl -X POST http://localhost:8199/api/embedding/search \
  -F query_image=@/宿主机/查询图片.jpg \
  -F index_path=/app/outputs/my_index/siglip2_base \
  -F top_k=10 -F exact=true \
  -F 'filters_json={"project_id":"project-a"}'

curl 'http://localhost:8199/api/embedding/index?index_path=/app/outputs/my_index/siglip2_base'

curl -X POST http://localhost:8199/api/embedding/upsert \
  -H 'Content-Type: application/json' \
  -d '{"index_path":"/app/outputs/my_index/siglip2_base","image_paths":["/app/data/照片目录/new.jpg"],"metadata":{"scene_id":"scene-1"}}'

curl -X POST http://localhost:8199/api/embedding/delete \
  -H 'Content-Type: application/json' \
  -d '{"index_path":"/app/outputs/my_index/siglip2_base","image_paths":["/app/data/照片目录/new.jpg"]}'
```

同一路径重复写入使用同一 ID，修改图片后重新 upsert 更新向量；未传入的过滤元数据保留。删除只删除向量记录。`metadata_json`、`filters_json` 和 `metadata` 支持 `project_id`、`play_id`、`scene_id` 三种非空字符串等值字段。写入有确认后可查询；多批增量写入不是跨批事务，失败后可幂等重试，应检查响应中的 `failed` 和 `errors`。全量重建则使用新集合，全部成功后才原子发布清单。

旧 `/api/face/build_database`、`/api/face/recognize` 接口沿用，自动选择配置后端。图像建库可传 `backend=faiss` 生成旧格式；需使用没有 Qdrant 清单的独立目录。Qdrant 暂不支持原 FAISS 的全图库重排/查询扩展参数，使用这些实验功能时保留 FAISS 索引。

## 显式迁移与回退

迁移前确认旧向量的模型版本。旧格式没有完整模型签名，工具要求 `--acknowledge-legacy-model` 并记录 `legacy_model_unverified=true`；该标记不是对原模型来源的验证。无法确认时从原图重新提取。

```bash
# FAISS → 新目录；原 .faiss/.npy/路径清单保持不变
docker compose exec theatrical-photo-sorting python tools/migrate_vector_store.py faiss \
  --source /app/outputs/old_index/siglip2_base \
  --destination /app/outputs/migrated_index/siglip2_base \
  --model siglip2_base --acknowledge-legacy-model \
  --report /app/outputs/migration_faiss.json

# 启动旧 ES 后迁移人脸记录，源索引不会被删除
docker compose --profile legacy up -d elasticsearch
docker compose exec theatrical-photo-sorting python tools/migrate_vector_store.py elasticsearch \
  --source face_database_512 --acknowledge-legacy-model \
  --report /app/outputs/migration_es.json
```

实际使用 `.env.docker` 的部署给以上命令追加 `--env-file .env.docker`（放在 `compose` 后）。FAISS 的 `.npy` 使用 mmap 分批读取；缺少 `.npy` 时需加载原 FAISS 索引后重建向量，仍会占用该索引的内存。ES 使用滚动读取，按人物和原图路径生成稳定 ID；重复源记录会合并。ES 迁移应在停止源库写入的维护窗口执行，计数校验不能证明并发更新期间内容完全一致。

回退图像检索时直接指定保留的旧 FAISS 目录。回退人脸后端时设置 `DOCKER_FACE_VECTOR_BACKEND=elasticsearch`，启动 `legacy` 服务并重建应用容器；Qdrant 中的新写入不会自动同步回旧 ES，切回前需明确数据版本。图像重建清单保留 `previous_collection` 供人工恢复，日常使用无需修改。

## 容量与备份

数据库可以部署到独立磁盘或服务器；默认命名卷仍在本机 Docker 数据目录，不能解决该磁盘本身空间不足。768 维 float32 每百万条原始向量约 2.86 GiB；旧格式同时保存 `.faiss` 和 `.npy` 约 5.72 GiB，再加路径及其他文件。Qdrant 还需要 HNSW、payload、WAL 与段文件，实际磁盘可能更多，详见实测。构建新一代时还需预留旧一代和临时构建空间。

监控卷剩余空间、数据库内存和集合状态；4 GiB 容器上限不是容量承诺。原图、连接清单及数据库快照都要备份；只备份 `vector_store.json` 不能恢复向量。部署到外部 Qdrant 时，配置对应地址及访问控制。此次单节点部署没有验证多机容灾。

全量重建保留历史集合，不会自动删除。可使用以下工具检查所有者位于指定根目录、至少 24 小时前创建的旧集合；它保留当前与上一代，默认仅列出候选：

```bash
docker compose exec theatrical-photo-sorting python tools/prune_vector_generations.py \
  --root /app/outputs/my_index
# 核对候选后加 --apply 执行。没有所有者标记/没有清单的集合不会被该工具删除。
```

被遗弃的索引目录、异常退出前未发布的集合和实验集合需按来源单独核对处理；此工具不作全库清空。

## 容器复测

`tests/test_vector_store.py` 连接真实 Qdrant，使用自己的临时集合；仓库测试目录未打入应用镜像，可复制至挂载的 `outputs/` 后执行。真实模型 HTTP 验收使用 `tools/check_qdrant_workflow.py --output /app/outputs/新验收目录`，重启后追加 `--after-restart` 检查持久化并清理临时人脸记录。图像验收集合保留供核查。

性能对比先在 GPU 应用容器执行 `python tools/benchmark_vector_store.py --prepare --output /app/outputs/基准目录`，再使用 `docker-compose.benchmark.yml` 的 `vector-benchmark` 服务测量。真实模型提取会额外占用 GPU 内存，应在空闲时执行。纯向量基准需旧 ES 服务在线：

```bash
docker compose --profile legacy up -d qdrant elasticsearch
docker compose -f docker-compose.yml -f docker-compose.benchmark.yml \
  run --rm --no-deps vector-benchmark --output /app/outputs/基准目录 --requests 100
```

基准合成集合为明确观察 HNSW 将 `full_scan_threshold` 降至 1000 KB，生产默认仍为 10000 KB；报告记录实际配置及未索引尾部，勿据单次合成测量推断真实识别准确率。
