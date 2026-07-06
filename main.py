#!/usr/bin/env python3
"""
鱼类物种识别系统 - 主入口文件
"""

import argparse
import sys
import os
from pathlib import Path
import shutil

def setup_environment():
    """设置运行环境"""
    print("=" * 60)
    print("鱼类物种识别系统 - 基于YOLOv8的两阶段训练")
    print("=" * 60)
    
    # 添加项目根目录到Python路径
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # 检查关键依赖
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"  CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("✗ PyTorch: 未安装")
        print("  请运行: pip install torch torchvision")
    
    try:
        from ultralytics import YOLO
        print("✓ Ultralytics YOLOv8: 可用")
    except ImportError:
        print("✗ Ultralytics: 未安装")
        print("  请运行: pip install ultralytics")
    
    try:
        import gradio as gr
        print("✓ Gradio: 可用")
    except ImportError:
        print("✗ Gradio: 未安装")
        print("  请运行: pip install gradio")
    
    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV: 未安装")
        print("  请运行: pip install opencv-python")
    
    return True

def check_and_prepare_data():
    """检查并准备数据"""
    print("\n📊 检查数据集...")
    
    # 检查原始数据是否存在
    if not os.path.exists("data/raw"):
        print("❌ 找不到原始数据: data/raw/")
        print("   请确保原始数据已放置正确")
        print("   数据目录结构应如下：")
        print("   data/raw/")
        print("   ├── 01. Anabas Testudineus/")
        print("   ├── 02. Batasio Tengana/")
        print("   └── ...")
        return False
    
    # 统计图像数量
    total_images = 0
    for i in range(1, 9):
        folder_name = f"{i:02d}. "  # 01. , 02. 等
        folder_path = os.path.join("data/raw", folder_name)
        # 因为文件夹名后面还有类别名，所以用模糊匹配
        for item in os.listdir("data/raw"):
            if item.startswith(folder_name):
                folder_path = os.path.join("data/raw", item)
                break
        
        if os.path.exists(folder_path):
            images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            print(f"   类别 {i}: {len(images)} 张图像")
            total_images += len(images)
    
    print(f"   总计: {total_images} 张图像")
    return True

def generate_smart_annotations():
    """生成智能边界框"""
    print("\n[1/3] 生成智能边界框...")
    
    try:
        from src.utils.smart_bbox_generator import SmartBBoxGenerator
        generator = SmartBBoxGenerator()
        count = generator.convert_dataset()
        
        if count > 0:
            print(f"\n✅ 生成成功! 共处理 {count} 张图像")
            return True
        else:
            print("❌ 生成失败")
            return False
            
    except Exception as e:
        print(f"❌ 生成边界框失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_model():
    """训练模型"""
    print("\n[2/3] 训练模型...")
    
    # 检查智能标注数据
    if not os.path.exists("data/smart_annotated/data.yaml"):
        print("❌ 找不到智能标注数据，请先生成智能边界框")
        return False
    
    # 运行两阶段训练
    try:
        from src.training.two_stage_trainer import TwoStageYOLOTrainer
        trainer = TwoStageYOLOTrainer()
        results = trainer.run_complete_pipeline()
        
        if results:
            # 复制最终模型到固定位置
            if results.get('stage2_model') and os.path.exists(results['stage2_model']):
                final_model_path = "models/final_fish_model.pt"
                os.makedirs("models", exist_ok=True)
                shutil.copy2(results['stage2_model'], final_model_path)
                print(f"\n✅ 训练成功!")
                print(f"最终模型已保存到: {final_model_path}")
                return True
        else:
            print("❌ 训练失败")
            return False
            
    except Exception as e:
        print(f"❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def launch_gui():
    """启动Web界面"""
    print("\n[3/3] 启动Web界面...")
    
    try:
        from gui.gradio_app import launch_app
        launch_app()
        return True
    except Exception as e:
        print(f"❌ 启动界面失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def cleanup_old_models():
    """清理旧的模型训练结果（可选）"""
    import glob
    old_patterns = [
        "fish_detection/fish_detection_*",
        "runs/detect/*",
        "results/train_logs/*"
    ]
    
    for pattern in old_patterns:
        for path in glob.glob(pattern):
            try:
                import shutil
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    print(f"🧹 清理: {path}")
            except:
                pass

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="鱼类物种识别系统 - 基于YOLOv8的两阶段训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py train             # 训练模型（两阶段训练）
  python main.py gui               # 启动Web界面
  python main.py all               # 完整流程：准备数据 -> 训练 -> 启动界面
  python main.py annotate          # 只生成智能边界框
  python main.py cleanup           # 清理旧的训练结果
        """
    )
    
    parser.add_argument('mode', choices=['train', 'gui', 'annotate', 'all', 'cleanup'], 
                       help='运行模式')
    parser.add_argument('--port', type=int, default=7860, help='Web界面端口')
    
    args = parser.parse_args()
    
    # 设置环境
    if not setup_environment():
        return
    
    if args.mode == 'train':
        # 检查数据
        if not check_and_prepare_data():
            return
        
        # 检查是否需要先生成智能边界框
        if not os.path.exists("data/smart_annotated/data.yaml"):
            print("⚠️ 未找到智能边界框数据，正在生成...")
            if not generate_smart_annotations():
                return
        
        # 训练模型
        train_model()
        
    elif args.mode == 'gui':
        # 启动Web界面
        launch_gui()
        
    elif args.mode == 'annotate':
        # 生成智能边界框
        if not check_and_prepare_data():
            return
        generate_smart_annotations()
        
    elif args.mode == 'all':
        # 完整流程
        print("\n🚀 运行完整流程...")
        
        # 1. 检查数据
        if not check_and_prepare_data():
            return
        
        # 2. 生成智能边界框
        if generate_smart_annotations():
            # 3. 训练模型
            if train_model():
                # 4. 启动界面
                launch_gui()
            else:
                print("❌ 训练失败，跳过界面启动")
        else:
            print("❌ 生成标注失败，流程中止")
    
    elif args.mode == 'cleanup':
        # 清理旧的训练结果
        print("\n🧹 清理旧的训练结果...")
        cleanup_old_models()
        print("✅ 清理完成")
    
    print("\n" + "=" * 60)
    print("程序执行完成")
    print("=" * 60)

if __name__ == "__main__":
    main()