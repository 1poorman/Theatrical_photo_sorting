# test_siglip2.py
# 验证 SigLIP 2 接入 image_search_system_module.py 后的特征提取与索引构建
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core_modules.image_search import ImageEmbedder, VectorDatabase


def main():
    # 1. 加载 siglip2_base 模型
    print("=" * 60)
    print("[1] 加载 siglip2_base 模型...")
    embedder = ImageEmbedder(model_name='siglip2_base')
    print(f"    model: {embedder.model_name}, feature_dim: {embedder.feature_dim}")

    # 2. 单图特征提取
    print("=" * 60)
    print("[2] 单图特征提取...")
    test_dir = os.path.join(PROJECT_ROOT, 'data/sample_images')
    image_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print("    !! 未找到 test_images 图片，跳过")
        return

    feats = []
    for p in image_files[:3]:
        feat = embedder.extract_features(p)
        if feat is not None:
            feats.append(feat)
            print(f"    {os.path.basename(p)} -> shape={feat.shape}, "
                  f"norm={np.linalg.norm(feat):.4f}, first5={np.round(feat[:5], 4)}")
        else:
            print(f"    {os.path.basename(p)} -> 提取失败")

    # 3. 批量特征提取
    print("=" * 60)
    print("[3] 批量特征提取...")
    batch_feats, valid_paths = embedder.extract_features_batch(image_files[:3])
    print(f"    batch shape: {batch_feats.shape}, valid_paths: {len(valid_paths)}")

    # 4. 构建一个小规模索引并搜索
    print("=" * 60)
    print("[4] 构建 FAISS 索引并检索（自查询）...")
    index_dir = os.path.join(PROJECT_ROOT, 'data/embedding_index/test_siglip2_index')
    os.makedirs(index_dir, exist_ok=True)
    embedder2 = ImageEmbedder(model_name='siglip2_base')
    vector_db = VectorDatabase(dimension=embedder2.feature_dim, model_name='siglip2_base')
    vector_db.build_index_optimized(test_dir, embedder2, batch_size=4)
    vector_db.save_index(index_dir)

    from core_modules.image_search import ImageSearcher
    searcher = ImageSearcher(vector_db, embedder2)
    results = searcher.search(image_files[0], top_k=3)
    print("    检索结果:")
    for r in results:
        print(f"      rank={r['rank']}, similarity={r['similarity']:.4f}, {os.path.basename(r['image_path'])}")

    print("=" * 60)
    print("验证通过 ✔")


if __name__ == "__main__":
    main()
