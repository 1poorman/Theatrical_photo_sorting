# main.py
import os
import sys
import argparse
from image_search_system import ImageEmbedder, VectorDatabase, ImageSearcher, display_results, create_results_image
import datetime

def build_index(image_folder, index_save_path, model_name='resnet50', use_optimized=True):
    """构建图片索引"""
    print(f"初始化图片嵌入器 (模型: {model_name})...")
    embedder = ImageEmbedder(model_name=model_name)
    
    # 根据模型设置特征维度
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
        dimension = 768  # nomic-embed-vision-v1.5的特征维度
    elif model_name.startswith('blip2'):
        dimension = 768  # BLIP2 image embedding dimension
    else:
        if model_name == 'resnet50' or model_name == 'resnet101':
            dimension = 2048
        else:  # resnet18
            dimension = 512
    
    print(f"初始化向量数据库 (维度: {dimension}, 模型: {model_name})...")
    vector_db = VectorDatabase(dimension=dimension, model_name=model_name)
    
    # 构建索引
    if use_optimized:
        print("使用优化版索引构建...")
        vector_db.build_index_optimized(image_folder, embedder)
    else:
        print("使用标准版索引构建...")
        vector_db.build_index(image_folder, embedder)
    
    # 保存索引和模型信息
    vector_db.save_index(index_save_path)
    
    print("索引构建完成！")
    
    return vector_db, embedder

def search_image(query_image_path, index_path, top_k=5, show_results=True, save_results=True):
    """搜索相似图片"""
    # 从索引中获取模型信息
    model_info_path = os.path.join(index_path, "model_info.txt")
    model_name = "resnet50"  # 默认模型
    
    if os.path.exists(model_info_path):
        with open(model_info_path, 'r') as f:
            for line in f:
                if line.startswith('model:'):
                    model_name = line.split(':')[1].strip()
    
    print(f"使用模型: {model_name}")
    
    # 初始化组件
    embedder = ImageEmbedder(model_name=model_name)
    vector_db = VectorDatabase()
    vector_db.load_index(index_path)
    searcher = ImageSearcher(vector_db, embedder)
    
    # 执行搜索
    if os.path.exists(query_image_path):
        results = searcher.search(query_image_path, top_k=top_k)
        
        print("\n搜索结果:")
        for result in results:
            print(f"排名: {result['rank']}, 相似度: {result['similarity']:.4f}")
            print(f"图片路径: {result['image_path']}\n")
        
        # 保存结果图片
        if save_results and results:
            # 创建结果保存目录
            results_dir = "search_results"
            os.makedirs(results_dir, exist_ok=True)
            
            # 生成文件名
            query_name = os.path.splitext(os.path.basename(query_image_path))[0]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(results_dir, f"result_{query_name}_{timestamp}.jpg")
            
            # 创建并保存结果图片
            create_results_image(query_image_path, results, save_path=save_path)
        
        # 可视化结果
        if show_results and results:
            display_results(query_image_path, results)
        
        return results
    else:
        print(f"查询图片不存在: {query_image_path}")
        return []

def interactive_search(index_path):
    """交互式搜索模式"""
    # 从索引中获取模型信息
    model_info_path = os.path.join(index_path, "model_info.txt")
    model_name = "resnet50"  # 默认模型
    
    if os.path.exists(model_info_path):
        with open(model_info_path, 'r') as f:
            for line in f:
                if line.startswith('model:'):
                    model_name = line.split(':')[1].strip()
    
    print(f"使用模型: {model_name}")
    
    embedder = ImageEmbedder(model_name=model_name)
    vector_db = VectorDatabase()
    vector_db.load_index(index_path)
    searcher = ImageSearcher(vector_db, embedder)
    
    print("进入交互式搜索模式，输入图片路径进行搜索，输入 'quit' 退出")
    
    while True:
        query_path = input("\n请输入查询图片路径: ").strip()
        
        if query_path.lower() in ['quit', 'exit', 'q']:
            break
        
        if not os.path.exists(query_path):
            print("图片不存在，请重新输入")
            continue
        
        try:
            results = searcher.search(query_path, top_k=5)
            
            print("\n搜索结果:")
            for result in results:
                print(f"排名: {result['rank']}, 相似度: {result['similarity']:.4f}")
                print(f"图片路径: {result['image_path']}")
            
            # 询问是否保存结果图片
            save = input("\n是否保存搜索结果图片？(y/n): ").strip().lower()
            if save == 'y':
                # 创建结果保存目录
                results_dir = "search_results"
                os.makedirs(results_dir, exist_ok=True)
                
                # 生成文件名
                query_name = os.path.splitext(os.path.basename(query_path))[0]
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(results_dir, f"result_{query_name}_{timestamp}.jpg")
                
                # 创建并保存结果图片
                create_results_image(query_path, results, save_path=save_path)
            
            # 询问是否显示图片
            show = input("\n是否显示搜索结果图片？(y/n): ").strip().lower()
            if show == 'y':
                display_results(query_path, results)
                
        except Exception as e:
            print(f"搜索过程中发生错误: {e}")

def demo():
    """演示模式 - 当直接运行 python main.py 时执行"""
    print("=" * 50)
    print("基于ResNet/DINOv2/Nomic/BLIP2的图片搜索系统")
    print("=" * 50)
    
    # 检查是否有示例图片文件夹
    example_images = "example_images"
    index_path = "vector_index"
    
    if not os.path.exists(example_images):
        print(f"未找到示例图片文件夹 '{example_images}'")
        print("请创建一个包含图片的文件夹，或使用以下命令运行:")
        print("  python main.py --mode build --image_folder <你的图片文件夹路径> --model <模型名称>")
        print("  可用模型: resnet18, resnet50, resnet101, dinov2_small, dinov2_base, dinov2_large, dinov2_giant, nomic_vision, blip2")
        print("  python main.py --mode search --query_image <查询图片路径>")
        return
    
    # 检查是否已构建索引
    if not os.path.exists(index_path):
        print("检测到示例图片文件夹，但尚未构建索引")
        print("请选择要使用的模型:")
        print("1. ResNet18 (轻量级，特征维度: 512)")
        print("2. Resnet50 (平衡，特征维度: 2048)")
        print("3. ResNet101 (更强大，特征维度: 2048)")
        print("4. DINOv2 Small (自监督，特征维度: 384)")
        print("5. DINOv2 Base (自监督，特征维度: 768)")
        print("6. DINOv2 Large (自监督，特征维度: 1024)")
        print("7. Nomic Vision (本地模型，特征维度: 768)")
        print("8. BLIP2 (多模态模型，特征维度: 768)")

        model_choice = input("请输入选择 (1-8, 默认2): ").strip()
        
        model_map = {
            '1': 'resnet18',
            '2': 'resnet50',
            '3': 'resnet101',
            '4': 'dinov2_small',
            '5': 'dinov2_base',
            '6': 'dinov2_large',
            '7': 'nomic_vision',
            '8': 'blip2'
        }
        
        model_name = model_map.get(model_choice, 'resnet50')
        
        build = input(f"是否使用 {model_name} 构建索引？(y/n): ").strip().lower()
        if build == 'y':
            build_index(example_images, index_path, model_name=model_name, use_optimized=True)
        else:
            return
    
    # 进入交互式搜索模式
    interactive_search(index_path)

def main():
    # 如果没有命令行参数，进入演示模式
    if len(sys.argv) == 1:
        demo()
        return
    
    # 否则，使用命令行参数解析
    parser = argparse.ArgumentParser(description='基于ResNet/DINOv2/Nomic/BLIP2的图片搜索系统')
    parser.add_argument('--mode', choices=['build', 'search', 'interactive'], 
                       required=False, help='运行模式: build-构建索引, search-搜索, interactive-交互式搜索')
    parser.add_argument('--image_folder', help='图片文件夹路径（用于构建索引）')
    parser.add_argument('--index_path', default='vector_index', help='索引保存/加载路径')
    parser.add_argument('--query_image', help='查询图片路径（用于搜索模式）')
    parser.add_argument('--model', default='resnet50', 
                       choices=['resnet18', 'resnet50', 'resnet101', 
                               'dinov2_small', 'dinov2_base', 'dinov2_large', 'dinov2_giant',
                               'nomic_vision','blip2'],
                       help='使用的模型类型')
    parser.add_argument('--top_k', type=int, default=5, help='返回的相似图片数量')
    parser.add_argument('--optimized', action='store_true', help='使用优化版索引构建')
    parser.add_argument('--no_save', action='store_true', help='不保存结果图片')
    parser.add_argument('--no_display', action='store_true', help='不显示结果图片')
    
    args = parser.parse_args()
    
    if args.mode == 'build':
        if not args.image_folder:
            print("错误: 构建索引模式需要指定 --image_folder 参数")
            return
        
        if not os.path.exists(args.image_folder):
            print(f"错误: 图片文件夹不存在: {args.image_folder}")
            return
        
        build_index(args.image_folder, args.index_path, args.model, args.optimized)
        
    elif args.mode == 'search':
        if not args.query_image:
            print("错误: 搜索模式需要指定 --query_image 参数")
            return
        
        search_image(
            args.query_image, 
            args.index_path, 
            args.top_k, 
            show_results=not args.no_display,
            save_results=not args.no_save
        )
        
    elif args.mode == 'interactive':
        if not os.path.exists(args.index_path):
            print(f"错误: 索引路径不存在: {args.index_path}")
            print("请先使用 build 模式构建索引")
            return
        
        interactive_search(args.index_path)

if __name__ == "__main__":
    main()