# 在项目根目录下执行
"""
智能边界框生成器 - 为分类数据集生成合理的边界框（修正版）
确保每张原始图像都有对应的标注，验证集每类约20张
"""

import os
import cv2
import numpy as np
from pathlib import Path
import shutil
import yaml
import random
import traceback
import json

class SmartBBoxGenerator:
    def __init__(self, raw_data_path="data/raw", output_path="data/smart_annotated"):
        self.raw_path = raw_data_path
        self.output_path = output_path
        
        self.classes = [
            "Anabas_Testudineus",
            "Batasio_Tengana", 
            "Channa_Punctata",
            "Heteropneustes_Fossilis",
            "Marcrobrachium_Malcoimsonii",
            "Mstacembelus_Armatus", 
            "Ompok_Bimaculatus",
            "Puntius"
        ]
        
        # 支持的图像扩展名
        self.image_extensions = [
            '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.jfif',
            '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF', '.WEBP', '.JFIF'
        ]
        
        # 验证集目标数量
        self.target_val_per_class = 20
        
        # 保存划分信息
        self.split_info_file = os.path.join(output_path, "split_info.json")
    
    def generate_simple_bbox(self, img_path, img=None):
        """生成简单的边界框 - 确保100%成功"""
        try:
            if img is None:
                img = cv2.imread(str(img_path))
                if img is None:
                    # 尝试用PIL读取
                    try:
                        from PIL import Image
                        pil_img = Image.open(img_path)
                        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                        pil_img.close()
                    except:
                        return [0.5, 0.5, 0.8, 0.8]  # 默认边界框
            
            h, w = img.shape[:2]
            
            # 方法1: 基于简单阈值分割
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
                
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # 合并所有轮廓
                    all_points = []
                    for contour in contours:
                        if cv2.contourArea(contour) > 100:
                            all_points.extend(contour.reshape(-1, 2))
                    
                    if all_points:
                        all_points = np.array(all_points)
                        x, y, bw, bh = cv2.boundingRect(all_points)
                        
                        # 计算边界框
                        x_center = (x + bw/2) / w
                        y_center = (y + bh/2) / h
                        bbox_width = bw / w
                        bbox_height = bh / h
                        
                        # 验证边界框合理性
                        if (0.05 <= bbox_width <= 0.95 and 
                            0.05 <= bbox_height <= 0.95 and
                            0 <= x_center <= 1 and 0 <= y_center <= 1):
                            return [x_center, y_center, bbox_width, bbox_height]
            except:
                pass
            
            # 方法2: 基于颜色差异
            try:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                # 创建简单的背景/前景分割
                mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 240]))
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    x, y, bw, bh = cv2.boundingRect(largest)
                    
                    x_center = (x + bw/2) / w
                    y_center = (y + bh/2) / h
                    bbox_width = bw / w
                    bbox_height = bh / h
                    
                    if (0.05 <= bbox_width <= 0.95 and 
                        0.05 <= bbox_height <= 0.95):
                        return [x_center, y_center, bbox_width, bbox_height]
            except:
                pass
            
        except Exception as e:
            print(f"   生成边界框时出错: {e}")
        
        # 默认边界框（图像中心，80%大小）
        return [0.5, 0.5, 0.8, 0.8]
    
    def _read_image(self, img_path):
        """读取图像，支持多种格式"""
        img = cv2.imread(str(img_path))
        if img is None:
            try:
                from PIL import Image
                pil_img = Image.open(img_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                pil_img.close()
            except:
                return None
        return img
    
    def convert_dataset(self, train_ratio=0.8):
        """转换整个数据集 - 确保每类验证集约20张"""
        print("=" * 60)
        print("🚀 开始智能边界框生成（目标验证集: 20张/类）")
        print("=" * 60)
        
        print(f"原始数据路径: {self.raw_path}")
        print(f"输出路径: {self.output_path}")
        
        # 清空现有输出目录（强制重新生成）
        if os.path.exists(self.output_path):
            print(f"🧹 清空现有输出目录: {self.output_path}")
            shutil.rmtree(self.output_path)
        
        # 创建输出目录
        for split in ["train", "val"]:
            os.makedirs(os.path.join(self.output_path, "images", split), exist_ok=True)
            os.makedirs(os.path.join(self.output_path, "labels", split), exist_ok=True)
        
        total_images = 0
        successful_images = 0
        failed_images = []
        
        # 保存划分信息
        split_info = {
            "val_images": [],  # 验证集图像路径列表
            "class_val_counts": {i: 0 for i in range(8)}
        }
        
        # 处理每个类别
        for class_idx in range(8):
            # 原始文件夹格式: "01. Anabas Testudineus"
            original_folder = f"{class_idx+1:02d}. {self.classes[class_idx].replace('_', ' ')}"
            class_path = os.path.join(self.raw_path, original_folder)
            
            if not os.path.exists(class_path):
                print(f"⚠️ 警告: {class_path} 不存在")
                continue
            
            # 获取所有图像文件
            image_files = []
            for img_file in Path(class_path).iterdir():
                if img_file.is_file():
                    ext = img_file.suffix.lower()
                    if ext in [ext.lower() for ext in self.image_extensions]:
                        image_files.append(img_file)
            
            # 去重
            image_files = list(set(image_files))
            
            if not image_files:
                print(f"⚠️ 警告: {class_path} 中没有图像文件")
                continue
            
            print(f"\n📁 处理类别 {self.classes[class_idx]}: {len(image_files)} 张图像")
            
            # 随机打乱并划分
            random.shuffle(image_files)
            
            # 计算验证集数量
            total_images_in_class = len(image_files)
            
            # 目标验证集数量：尽量接近20，但不超过总图像的30%
            target_val_count = self.target_val_per_class
            
            # 确保验证集不超过总图像的30%
            max_val_by_ratio = int(total_images_in_class * 0.3)
            target_val_count = min(target_val_count, max_val_by_ratio)
            
            # 确保验证集不超过总图像数-1（至少留1张训练）
            target_val_count = min(target_val_count, total_images_in_class - 1)
            
            # 确保至少1张验证图像
            target_val_count = max(1, target_val_count)
            
            # 对于样本特别少的类别，调整策略
            if total_images_in_class < 30:
                # 小样本类别：按25%比例
                target_val_count = int(total_images_in_class * 0.25)
                target_val_count = max(1, min(target_val_count, 15))  # 最多15张
                print(f"  小样本类别，调整验证集数量为: {target_val_count}")
            
            # 计算划分点
            split_idx = total_images_in_class - target_val_count
            
            # 确保至少有5张训练图像（如果可能）
            if split_idx < 5 and total_images_in_class > 5:
                split_idx = 5
                target_val_count = total_images_in_class - split_idx
            
            train_files = image_files[:split_idx]
            val_files = image_files[split_idx:]
            
            print(f"  训练集: {len(train_files)} 张, 验证集: {len(val_files)} 张")
            
            # 处理训练集
            train_success = 0
            for img_file in train_files:
                if self._process_single_image_forced(img_file, class_idx, "train"):
                    train_success += 1
                else:
                    failed_images.append(str(img_file))
                total_images += 1
            
            # 处理验证集
            val_success = 0
            for img_file in val_files:
                if self._process_single_image_forced(img_file, class_idx, "val"):
                    val_success += 1
                    # 保存验证集图像信息
                    img_name = img_file.name
                    split_info["val_images"].append({
                        "class_id": class_idx,
                        "class_name": self.classes[class_idx],
                        "image_name": img_name,
                        "original_path": str(img_file)
                    })
                    split_info["class_val_counts"][class_idx] += 1
                else:
                    failed_images.append(str(img_file))
                total_images += 1
            
            successful_images += train_success + val_success
            print(f"  训练成功: {train_success}/{len(train_files)}, 验证成功: {val_success}/{len(val_files)}")
        
        # 保存划分信息
        with open(self.split_info_file, 'w', encoding='utf-8') as f:
            json.dump(split_info, f, ensure_ascii=False, indent=2)
        
        # 输出统计信息
        print(f"\n" + "=" * 60)
        print("✅ 转换完成!")
        print(f"📊 详细统计:")
        print(f"   总图像数: {total_images}")
        print(f"   成功处理: {successful_images}")
        print(f"   失败: {len(failed_images)}")
        print(f"   成功率: {successful_images/total_images*100:.2f}%")
        
        if failed_images:
            print(f"\n❌ 失败的图像 (前10张):")
            for img_path in failed_images[:10]:
                print(f"   {os.path.basename(img_path)}")
            if len(failed_images) > 10:
                print(f"   ... 还有 {len(failed_images)-10} 张")
        
        # 创建data.yaml文件
        self._create_data_yaml()
        
        # 输出验证集分布
        self._print_validation_distribution(split_info)
        
        return successful_images
    
    def _process_single_image_forced(self, img_file, class_idx, split):
        """强制处理单张图像 - 确保100%成功"""
        try:
            img_path = str(img_file)
            img_name = img_file.name
            
            # 目标路径
            img_dst = os.path.join(self.output_path, "images", split, img_name)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(self.output_path, "labels", split, label_name)
            
            # 确保目录存在
            os.makedirs(os.path.dirname(img_dst), exist_ok=True)
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            
            # 1. 尝试读取图像
            img = self._read_image(img_path)
            
            # 2. 生成边界框
            bbox = self.generate_simple_bbox(img_path, img)
            
            # 3. 边界框验证和修正
            if bbox:
                bbox[0] = max(0.0, min(1.0, bbox[0]))  # x_center
                bbox[1] = max(0.0, min(1.0, bbox[1]))  # y_center
                bbox[2] = max(0.05, min(0.95, bbox[2]))  # width
                bbox[3] = max(0.05, min(0.95, bbox[3]))  # height
            else:
                bbox = [0.5, 0.5, 0.8, 0.8]  # 默认边界框
            
            # 4. 复制图像文件（强制复制）
            try:
                if os.path.exists(img_path):
                    shutil.copy2(img_path, img_dst)
                else:
                    print(f"❌ 源文件不存在: {img_path}")
                    return False
            except Exception as copy_error:
                print(f"❌ 复制失败 {img_name}: {copy_error}")
                return False
            
            # 5. 写入标签文件
            try:
                with open(label_path, 'w') as f:
                    f.write(f"{class_idx} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
            except Exception as write_error:
                print(f"❌ 写入标签失败 {img_name}: {write_error}")
                # 删除可能已复制的图像
                if os.path.exists(img_dst):
                    os.remove(img_dst)
                return False
            
            # 6. 验证文件是否成功创建
            if not os.path.exists(img_dst):
                print(f"❌ 图像文件未创建: {img_dst}")
                return False
            
            if not os.path.exists(label_path):
                print(f"❌ 标签文件未创建: {label_path}")
                if os.path.exists(img_dst):
                    os.remove(img_dst)
                return False
            
            # 验证标签内容
            try:
                with open(label_path, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        print(f"❌ 标签文件为空: {label_path}")
                        os.remove(label_path)
                        if os.path.exists(img_dst):
                            os.remove(img_dst)
                        return False
            except:
                pass
            
            return True
            
        except Exception as e:
            print(f"❌ 处理失败 {os.path.basename(str(img_file))}: {e}")
            traceback.print_exc()
            return False
    
    def _create_data_yaml(self):
        """创建YOLO数据配置文件"""
        data_yaml = {
            "path": os.path.abspath(self.output_path),
            "train": "images/train",
            "val": "images/val",
            "nc": len(self.classes),
            "names": {i: name for i, name in enumerate(self.classes)}
        }
        
        yaml_path = os.path.join(self.output_path, "data.yaml")
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"📄 数据配置文件: {yaml_path}")
    
    def _print_validation_distribution(self, split_info):
        """打印验证集分布"""
        print(f"\n📊 验证集类别分布:")
        for class_idx in range(8):
            count = split_info["class_val_counts"][class_idx]
            print(f"  {self.classes[class_idx]}: {count} 张")
    
    def get_split_info(self):
        """获取划分信息"""
        if os.path.exists(self.split_info_file):
            with open(self.split_info_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

# 使用示例
if __name__ == "__main__":
    generator = SmartBBoxGenerator(
        raw_data_path="data/raw",
        output_path="data/smart_annotated"
    )
    
    # 先备份现有数据
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = f"data/smart_annotated_backup_{timestamp}"
    
    if os.path.exists("data/smart_annotated"):
        print(f"📦 备份现有数据到: {backup_dir}")
        shutil.copytree("data/smart_annotated", backup_dir)
    
    # 重新生成数据
    generator.convert_dataset()