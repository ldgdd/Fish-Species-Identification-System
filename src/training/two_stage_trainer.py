"""
两阶段YOLOv8训练器 - 专门解决无边界框数据集问题（修正版）
保持验证集与第一阶段一致，每类约20张验证图像
"""

import os
import sys
import yaml
import json
import random  # 添加这行
from ultralytics import YOLO
from datetime import datetime
import torch
import numpy as np
from pathlib import Path
import shutil
import cv2
import glob

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

try:
    from src.training.train_config import TrainingConfig
except ImportError:
    from train_config import TrainingConfig

class TwoStageYOLOTrainer:
    """改进的两阶段训练器 - 保持验证集一致性"""
    
    def __init__(self):
        self.stage1_model = None
        self.stage2_model = None
        self.class_names = [
            "Anabas_Testudineus",
            "Batasio_Tengana", 
            "Channa_Punctata",
            "Heteropneustes_Fossilis",
            "Marcrobrachium_Malcoimsonii",
            "Mstacembelus_Armatus", 
            "Ompok_Bimaculatus",
            "Puntius"
        ]
        
        # 验证集划分信息
        self.split_info = None
        self.split_info_file = "data/smart_annotated/split_info.json"
        
    def load_split_info(self):
        """加载验证集划分信息"""
        if os.path.exists(self.split_info_file):
            with open(self.split_info_file, 'r', encoding='utf-8') as f:
                self.split_info = json.load(f)
            print(f"✅ 加载验证集划分信息，共 {len(self.split_info['val_images'])} 张验证图像")
            return True
        else:
            print(f"⚠️  找不到划分信息文件: {self.split_info_file}")
            return False
    
    def load_local_model(self, model_name):
        """加载本地模型"""
        # 检查常见路径
        possible_paths = [
            f"{model_name}.pt",
            f"D:/大三/鱼类物种识别系统/{model_name}.pt",
            f"models/{model_name}.pt",
            f"D:/大三/鱼类物种识别系统/models/{model_name}.pt",
        ]
        
        # 检查通配符路径
        wildcard_paths = [
            f"stage2_training/stage2_refined_*/weights/{model_name}.pt",
            f"stage1_training/stage1_baseline_*/weights/{model_name}.pt",
            f"results/train_logs/fish_detection/*/weights/{model_name}.pt"
        ]
        
        for wildcard in wildcard_paths:
            matches = glob.glob(wildcard, recursive=True)
            if matches:
                # 按修改时间排序，取最新的
                matches.sort(key=os.path.getmtime, reverse=True)
                return matches[0]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ 找到本地模型: {path}")
                return path
        
        # 如果没找到，尝试下载
        print(f"⚠️  未找到本地模型 {model_name}.pt，将尝试下载...")
        return None
    
    def analyze_class_distribution(self, data_path="data/raw"):
        """分析类别分布"""
        print("=" * 60)
        print("分析类别分布...")
        print("=" * 60)
        
        class_counts = {}
        
        for class_idx in range(8):
            folder_name = f"{class_idx+1:02d}. {self.class_names[class_idx].replace('_', ' ')}"
            class_path = os.path.join(data_path, folder_name)
            
            if os.path.exists(class_path):
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    image_files.extend(list(Path(class_path).glob(ext)))
                
                class_counts[self.class_names[class_idx]] = len(image_files)
                print(f"  {self.class_names[class_idx]}: {len(image_files)} 张图像")
            else:
                class_counts[self.class_names[class_idx]] = 0
                print(f"  {self.class_names[class_idx]}: 0 张图像")
        
        return class_counts
    
    def run_stage1(self):
        """第一阶段：使用智能边界框训练基础模型"""
        print("=" * 60)
        print("第一阶段：基础模型训练")
        print("=" * 60)
        
        # 分析类别分布
        self.analyze_class_distribution("data/raw")
        
        # 检查智能边界框数据是否存在
        if not os.path.exists("data/smart_annotated/data.yaml"):
            print("❌ 找不到智能边界框数据，请先运行 smart_bbox_generator.py")
            return None
        
        # 创建第一阶段配置
        config = TrainingConfig(
            data_yaml="data/smart_annotated/data.yaml",
            model_name="yolov8n",
            epochs=200,  # 增加训练轮数
            batch_size=16,
            img_size=640,
            device="0" if torch.cuda.is_available() else "cpu",
            name=f"stage1_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            lr0=0.001,
            mosaic=0.6,
            mixup=0.0,
            copy_paste=0.0,  # 第一阶段不使用复制粘贴增强
            patience=60  # 增加早停耐心
        )
        
        print("📋 第一阶段训练参数:")
        print(f"   模型: {config.model_name}")
        print(f"   Epochs: {config.epochs}")
        print(f"   数据: {config.data_yaml}")
        print(f"   早停耐心值: {config.patience}")
        
        # 尝试加载本地模型
        model_path = self.load_local_model(config.model_name)
        if model_path:
            print(f"📥 加载本地模型: {model_path}")
            model = YOLO(model_path)
        else:
            print("🔄 使用预训练模型...")
            model = YOLO(f"{config.model_name}.pt")
        
        # 训练模型
        results = model.train(
            data=config.data_yaml,
            epochs=config.epochs,
            batch=config.batch_size,
            imgsz=config.img_size,
            device=config.device,
            project="stage1_training",
            name=config.name,
            exist_ok=True,
            save=True,
            save_period=10,
            patience=config.patience,
            lr0=config.lr0,
            lrf=config.lrf,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            hsv_h=config.hsv_h,
            hsv_s=config.hsv_s,
            hsv_v=config.hsv_v,
            degrees=config.degrees,
            translate=config.translate,
            scale=config.scale,
            shear=config.shear,
            fliplr=config.fliplr,
            mosaic=config.mosaic,
            mixup=config.mixup,
            copy_paste=config.copy_paste,
            verbose=True,
            plots=True,
            val=True,
            cos_lr=config.cos_lr,
            close_mosaic=config.close_mosaic
        )
        
        # 保存第一阶段模型路径
        stage1_dir = f"stage1_training/{config.name}"
        self.stage1_model = f"{stage1_dir}/weights/best.pt"
        
        print(f"\n✅ 第一阶段训练完成!")
        print(f"📁 模型保存到: {self.stage1_model}")
        
        # 验证模型
        if os.path.exists(self.stage1_model):
            val_model = YOLO(self.stage1_model)
            val_results = val_model.val(
                data=config.data_yaml,
                split="val",
                plots=True,
                save_json=True
            )
        
        return self.stage1_model
    
    def refine_annotations_with_consistent_val(self, model_path=None, conf_threshold=0.25):
        """重新标注：保持验证集与第一阶段完全一致"""
        print("=" * 60)
        print("重新标注（保持验证集一致性）...")
        print("=" * 60)
        
        # 加载验证集划分信息
        if not self.load_split_info():
            print("❌ 无法加载验证集划分信息，将使用标准划分")
            return self.refine_annotations_enhanced(model_path, conf_threshold)
        
        if model_path is None and self.stage1_model:
            model_path = self.stage1_model
        
        if model_path is None or not os.path.exists(model_path):
            print("❌ 找不到第一阶段模型")
            return None
        
        # 加载第一阶段模型
        model = YOLO(model_path)
        
        output_path = "data/refined_annotated_enhanced"
        
        # 创建输出目录
        for split in ["train", "val"]:
            os.makedirs(os.path.join(output_path, "images", split), exist_ok=True)
            os.makedirs(os.path.join(output_path, "labels", split), exist_ok=True)
        
        total_images = 0
        successful_refinements = 0
        failed_images = []
        
        # 构建验证集图像名称集合（便于快速查找）
        val_image_set = set()
        for val_info in self.split_info["val_images"]:
            val_image_set.add(val_info["image_name"])
        
        print(f"📊 验证集图像数量: {len(val_image_set)}")
        
        # 处理每个类别
        for class_idx, class_name in enumerate(self.class_names):
            original_folder = f"{class_idx+1:02d}. {class_name.replace('_', ' ')}"
            class_path = os.path.join("data/raw", original_folder)
            
            if not os.path.exists(class_path):
                print(f"⚠️  类别路径不存在: {class_path}")
                continue
            
            # 获取所有图像
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(list(Path(class_path).glob(ext)))
            
            if not image_files:
                print(f"⚠️  类别 {class_name} 没有图像")
                continue
            
            print(f"🔄 重新标注类别 {class_name}: {len(image_files)} 张图像")
            
            # 根据划分信息分配训练集和验证集
            train_files = []
            val_files = []
            
            for img_file in image_files:
                img_name = img_file.name
                if img_name in val_image_set:
                    val_files.append(img_file)
                else:
                    train_files.append(img_file)
            
            # 如果划分信息不完整，按比例分配
            if not val_files:
                print(f"⚠️  类别 {class_name} 在划分信息中未找到验证图像，按比例划分...")
                random.shuffle(image_files)
                split_idx = int(0.8 * len(image_files))
                train_files = image_files[:split_idx]
                val_files = image_files[split_idx:]
            
            print(f"  训练集: {len(train_files)} 张, 验证集: {len(val_files)} 张")
            
            # 处理训练集
            for img_file in train_files:
                if self._refine_single_image_enhanced(
                    img_file, class_idx, model, output_path, "train", 
                    conf_threshold, class_name
                ):
                    successful_refinements += 1
                else:
                    failed_images.append(str(img_file))
                total_images += 1
            
            # 处理验证集
            for img_file in val_files:
                if self._refine_single_image_enhanced(
                    img_file, class_idx, model, output_path, "val",
                    conf_threshold, class_name
                ):
                    successful_refinements += 1
                else:
                    failed_images.append(str(img_file))
                total_images += 1
        
        # 创建data.yaml
        data_yaml = {
            "path": os.path.abspath(output_path),
            "train": "images/train",
            "val": "images/val",
            "nc": len(self.class_names),
            "names": {i: name for i, name in enumerate(self.class_names)}
        }
        
        yaml_path = os.path.join(output_path, "data.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        # 输出统计信息
        print(f"\n✅ 重新标注完成!")
        print(f"📊 统计:")
        print(f"   总图像数: {total_images}")
        print(f"   成功重新标注: {successful_refinements}")
        print(f"   失败图像数: {len(failed_images)}")
        print(f"   成功率: {successful_refinements/total_images*100:.2f}%")
        print(f"   输出路径: {output_path}")
        
        # 输出验证集分布
        self._print_val_distribution(output_path)
        
        return output_path
    
    def _refine_single_image_enhanced(self, img_file, true_class_idx, model, 
                                     output_path, split, conf_threshold, class_name):
        """增强的单图像重新标注"""
        try:
            img_path = str(img_file)
            
            # 读取图像
            img = cv2.imread(img_path)
            if img is None:
                print(f"❌ 无法读取图像: {img_path}")
                return False
            
            height, width = img.shape[:2]
            
            # 尝试多种检测策略
            best_bbox = None
            best_conf = 0
            
            # 策略1: 标准检测
            results = model(img_path, conf=conf_threshold, verbose=False)
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        # 转换到YOLO格式
                        x_center = (x1 + x2) / (2 * width)
                        y_center = (y1 + y2) / (2 * height)
                        bbox_width = (x2 - x1) / width
                        bbox_height = (y2 - y1) / height
                        
                        if conf > best_conf:
                            best_conf = conf
                            best_bbox = [x_center, y_center, bbox_width, bbox_height]
            
            # 策略2: 降低阈值检测（针对低性能类别）
            if best_bbox is None and class_name in ["Anabas_Testudineus", 
                                                    "Heteropneustes_Fossilis", 
                                                    "Mstacembelus_Armatus"]:
                results_low = model(img_path, conf=0.1, verbose=False)
                for result in results_low:
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes
                        for i, box in enumerate(boxes):
                            cls = int(box.cls[0].cpu().numpy())
                            if cls == true_class_idx:  # 只考虑正确类别的检测
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = box.conf[0].cpu().numpy()
                                
                                x_center = (x1 + x2) / (2 * width)
                                y_center = (y1 + y2) / (2 * height)
                                bbox_width = (x2 - x1) / width
                                bbox_height = (y2 - y1) / height
                                
                                if conf > best_conf:
                                    best_conf = conf
                                    best_bbox = [x_center, y_center, bbox_width, bbox_height]
            
            # 策略3: 从智能标注中读取
            if best_bbox is None:
                smart_label_path = img_path.replace("data/raw", "data/smart_annotated") \
                                          .replace("images", "labels") \
                                          .replace(".jpg", ".txt") \
                                          .replace(".jpeg", ".txt") \
                                          .replace(".png", ".txt")
                
                if os.path.exists(smart_label_path):
                    with open(smart_label_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            line = line.strip()
                            if line:
                                parts = line.split()
                                if len(parts) >= 5:
                                    cls_id = int(parts[0])
                                    if cls_id == true_class_idx:
                                        best_bbox = list(map(float, parts[1:5]))
                                        best_conf = 0.5  # 中等置信度
                                        break
            
            # 策略4: 使用自适应边界框
            if best_bbox is None:
                # 基于图像特征生成边界框
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # 找到最大的轮廓
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    x_center = (x + w/2) / width
                    y_center = (y + h/2) / height
                    bbox_width = w / width * 0.8  # 稍微缩小
                    bbox_height = h / height * 0.8
                    
                    best_bbox = [x_center, y_center, bbox_width, bbox_height]
                    best_conf = 0.3
                else:
                    # 使用默认边界框
                    best_bbox = [0.5, 0.5, 0.7, 0.7]
                    best_conf = 0.1
            
            # 确保边界框有效
            if best_bbox:
                # 边界框坐标约束
                best_bbox[0] = max(0.0, min(1.0, best_bbox[0]))
                best_bbox[1] = max(0.0, min(1.0, best_bbox[1]))
                best_bbox[2] = max(0.05, min(0.95, best_bbox[2]))  # 最小5%，最大95%
                best_bbox[3] = max(0.05, min(0.95, best_bbox[3]))
            
            # 复制图像
            img_name = os.path.basename(img_path)
            img_dst = os.path.join(output_path, "images", split, img_name)
            shutil.copy2(img_path, img_dst)
            
            # 保存标注
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(output_path, "labels", split, label_name)
            
            with open(label_path, 'w') as f:
                f.write(f"{true_class_idx} {best_bbox[0]:.6f} {best_bbox[1]:.6f} "
                       f"{best_bbox[2]:.6f} {best_bbox[3]:.6f}\n")
            
            if best_conf < 0.3:
                print(f"⚠️  低置信度标注: {img_name} (置信度: {best_conf:.2f})")
            
            return True
            
        except Exception as e:
            print(f"❌ 重新标注 {os.path.basename(str(img_file))} 失败: {e}")
            return False
    
    def _print_val_distribution(self, data_path):
        """打印验证集分布"""
        print(f"\n📊 验证集类别分布:")
        val_label_dir = Path(data_path) / "labels" / "val"
        if val_label_dir.exists():
            class_counts = {i: 0 for i in range(8)}
            for label_file in val_label_dir.glob("*.txt"):
                with open(label_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        class_id = int(content.split()[0])
                        class_counts[class_id] += 1
            
            for i in range(8):
                print(f"  {self.class_names[i]}: {class_counts[i]} 张")
    
    def refine_annotations_enhanced(self, model_path=None, conf_threshold=0.25):
        """增强的重新标注（备选方法）"""
        # 这是一个备选方法，如果主方法失败时使用
        print("=" * 60)
        print("增强的重新标注（备选方法）...")
        print("=" * 60)
        
        # 这里可以调用原来的增强标注逻辑
        # 为了简化，这里直接调用主方法
        return self.refine_annotations_with_consistent_val(model_path, conf_threshold)
    
    def run_stage2(self, refined_data_path="data/refined_annotated_enhanced"):
        """第二阶段：使用增强的重新标注数据训练精确模型"""
        print("=" * 60)
        print("第二阶段：精确模型训练（增强版）")
        print("=" * 60)
        
        # 检查重新标注的数据
        data_yaml_path = os.path.join(refined_data_path, "data.yaml")
        if not os.path.exists(data_yaml_path):
            print(f"❌ 找不到重新标注的数据: {data_yaml_path}")
            # 尝试其他可能的数据路径
            alt_paths = ["data/refined_annotated", "data/smart_annotated"]
            for alt_path in alt_paths:
                alt_yaml = os.path.join(alt_path, "data.yaml")
                if os.path.exists(alt_yaml):
                    refined_data_path = alt_path
                    data_yaml_path = alt_yaml
                    print(f"🔄 使用备选数据路径: {refined_data_path}")
                    break
        
        if not os.path.exists(data_yaml_path):
            print("❌ 找不到任何有效的数据文件")
            return None
        
        # 创建第二阶段配置（使用增强参数）
        config = TrainingConfig(
            data_yaml=data_yaml_path,
            model_name="yolov8m",
            epochs=400,  # 增加训练轮数
            batch_size=8,
            img_size=640,
            device="0" if torch.cuda.is_available() else "cpu",
            name=f"stage2_refined_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            lr0=0.0002,  # 更低的学习率
            mosaic=0.7,   # 增加数据增强
            mixup=0.3,    # 增加mixup
            copy_paste=0.2,  # 使用复制粘贴增强
            patience=120  # 大幅增加早停耐心值
        )
        
        print("📋 第二阶段训练参数:")
        print(f"   模型: {config.model_name}")
        print(f"   Epochs: {config.epochs}")
        print(f"   数据: {config.data_yaml}")
        print(f"   早停耐心值: {config.patience}")
        print(f"   类别权重: {config.class_weights}")
        
        # 优先使用第一阶段模型
        if self.stage1_model and os.path.exists(self.stage1_model):
            print(f"📥 从第一阶段继承模型: {self.stage1_model}")
            model = YOLO(self.stage1_model)
        else:
            # 尝试加载本地模型
            model_path = self.load_local_model("yolov8m")
            if model_path:
                print(f"📥 加载本地模型: {model_path}")
                model = YOLO(model_path)
            else:
                print("🔄 使用yolov8n预训练模型...")
                model = YOLO("yolov8n.pt")
        
        # 训练模型（使用增强的训练参数）
        results = model.train(
            data=config.data_yaml,
            epochs=config.epochs,
            batch=config.batch_size,
            imgsz=config.img_size,
            device=config.device,
            project="stage2_training",
            name=config.name,
            exist_ok=True,
            save=True,
            save_period=10,
            patience=config.patience,
            lr0=config.lr0,
            lrf=config.lrf,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            hsv_h=config.hsv_h,
            hsv_s=config.hsv_s,
            hsv_v=config.hsv_v,
            degrees=config.degrees,
            translate=config.translate,
            scale=config.scale,
            shear=config.shear,
            flipud=config.flipud,
            fliplr=config.fliplr,
            mosaic=config.mosaic,
            mixup=config.mixup,
            copy_paste=config.copy_paste,
            perspective=config.perspective,
            cos_lr=config.cos_lr,
            close_mosaic=config.close_mosaic,
            verbose=True,
            plots=True,
            val=True
        )
        
        # 保存第二阶段模型路径
        stage2_dir = f"stage2_training/{config.name}"
        self.stage2_model = f"{stage2_dir}/weights/best.pt"
        
        print(f"\n✅ 第二阶段训练完成!")
        print(f"📁 最终模型保存到: {self.stage2_model}")
        
        # 最终验证
        if os.path.exists(self.stage2_model):
            final_model = YOLO(self.stage2_model)
            final_results = final_model.val(
                data=config.data_yaml,
                split="val",
                plots=True,
                save_json=True,
                conf=0.001,  # 低置信度阈值以检测困难样本
                iou=0.6
            )
            
            # 打印详细的类别性能
            print("\n📊 最终模型性能:")
            if hasattr(final_results, 'results_dict'):
                results_dict = final_results.results_dict
                print(f"   mAP50: {results_dict.get('metrics/mAP50(B)', 0):.3f}")
                print(f"   mAP50-95: {results_dict.get('metrics/mAP50-95(B)', 0):.3f}")
            
        return self.stage2_model
    
    def run_complete_pipeline(self):
        """运行完整的增强训练流程"""
        print("=" * 60)
        print("开始完整的两阶段训练流程（修正版）")
        print("=" * 60)
        
        results = {}
        
        # 第一阶段
        print("\n🚀 开始第一阶段训练...")
        stage1_model = self.run_stage1()
        if stage1_model and os.path.exists(stage1_model):
            results['stage1_model'] = stage1_model
            
            # 重新标注（保持验证集一致）
            print("\n🔄 开始保持验证集一致的重新标注...")
            refined_data = self.refine_annotations_with_consistent_val(stage1_model, conf_threshold=0.25)
            if refined_data:
                results['refined_data'] = refined_data
                
                # 第二阶段
                print("\n🚀 开始第二阶段训练...")
                stage2_model = self.run_stage2(refined_data)
                if stage2_model:
                    results['stage2_model'] = stage2_model
            else:
                print("❌ 重新标注失败，使用原始数据进行第二阶段训练")
                stage2_model = self.run_stage2("data/smart_annotated")
                if stage2_model:
                    results['stage2_model'] = stage2_model
        else:
            print("❌ 第一阶段训练失败，直接进行第二阶段训练")
            stage2_model = self.run_stage2()
            if stage2_model:
                results['stage2_model'] = stage2_model
        
        print("\n" + "=" * 60)
        print("训练流程完成!")
        print("=" * 60)
        
        if results:
            print("\n📊 训练结果摘要:")
            for key, value in results.items():
                print(f"  {key}: {value}")
            
            # 导出最终模型
            if 'stage2_model' in results and os.path.exists(results['stage2_model']):
                try:
                    model = YOLO(results['stage2_model'])
                    export_path = model.export(format="onnx")
                    print(f"\n📦 模型已导出为ONNX格式: {export_path}")
                except Exception as e:
                    print(f"\n⚠️ 导出ONNX失败: {e}")
                    print("但模型训练已完成，可以使用.pt文件")
        
        return results

# 主函数
def main():
    trainer = TwoStageYOLOTrainer()
    results = trainer.run_complete_pipeline()
    
    return results

if __name__ == "__main__":
    main()