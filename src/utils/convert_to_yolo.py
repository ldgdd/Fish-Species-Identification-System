import os
import cv2
import json
import shutil
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

class BDFreshFishToYOLO:
    def __init__(self, raw_data_path="data/raw", output_path="data/processed"):
        self.raw_path = raw_data_path
        self.output_path = output_path
        self.classes = [
            "01. Anabas Testudineus",
            "02. Batasio Tengana", 
            "03. Channa Punctata",
            "04. Heteropneustes Fossilis",
            "05. Marcrobrachium Malcoimsonii",
            "06. Mstacembelus Armatus",
            "07. Ompok Bimaculatus",
            "Punitus_Sophore"
        ]
        
    def create_annotations(self):
        """
        为BDFreshFish创建YOLO格式的标注
        由于原数据集没有边界框，我们可以使用整张图像作为边界框
        或者使用目标检测模型预标注（这里使用第一种方法）
        """
        annotations = []
        
        # 假设每个类别的图像在各自的文件夹中
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.raw_path, class_name)
            if not os.path.exists(class_path):
                print(f"警告: {class_path} 不存在")
                continue
                
            for img_file in os.listdir(class_path):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_path, img_file)
                    
                    # 读取图像尺寸
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                        
                    height, width = img.shape[:2]
                    
                    # 创建整图边界框 (x_center, y_center, width, height) 归一化
                    # 使用整张图像作为边界框
                    x_center = 0.5
                    y_center = 0.5
                    bbox_width = 1.0
                    bbox_height = 1.0
                    
                    annotation = {
                        "image_path": img_path,
                        "class_id": class_idx,
                        "bbox": [x_center, y_center, bbox_width, bbox_height],
                        "image_size": [width, height]
                    }
                    annotations.append(annotation)
                    
        return annotations
    
    def split_dataset(self, annotations, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """划分数据集"""
        # 按类别分层抽样
        class_annotations = {}
        for ann in annotations:
            class_id = ann["class_id"]
            if class_id not in class_annotations:
                class_annotations[class_id] = []
            class_annotations[class_id].append(ann)
            
        train_set, val_set, test_set = [], [], []
        
        for class_id, anns in class_annotations.items():
            # 随机打乱
            np.random.shuffle(anns)
            
            n_total = len(anns)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            train_set.extend(anns[:n_train])
            val_set.extend(anns[n_train:n_train+n_val])
            test_set.extend(anns[n_train+n_val:])
            
        return train_set, val_set, test_set
    
    def save_yolo_format(self, dataset, dataset_type):
        """保存为YOLO格式"""
        images_dir = os.path.join(self.output_path, "images", dataset_type)
        labels_dir = os.path.join(self.output_path, "labels", dataset_type)
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        for ann in dataset:
            # 复制图像文件
            src_img = ann["image_path"]
            img_name = os.path.basename(src_img)
            dst_img = os.path.join(images_dir, img_name)
            shutil.copy2(src_img, dst_img)
            
            # 创建标签文件
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_name)
            
            with open(label_path, 'w') as f:
                class_id = ann["class_id"]
                bbox = ann["bbox"]
                line = f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}"
                f.write(line + "\n")
    
    def create_data_yaml(self):
        """创建YOLO数据配置文件"""
        data_yaml = {
            "path": os.path.abspath(self.output_path),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(self.classes),
            "names": self.classes
        }
        
        yaml_path = os.path.join(self.output_path, "data.yaml")
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
            
        print(f"数据配置文件已保存: {yaml_path}")
        
    def convert(self):
        """执行转换"""
        print("开始转换数据集为YOLO格式...")
        
        # 创建标注
        annotations = self.create_annotations()
        print(f"总共找到 {len(annotations)} 张图像")
        
        # 划分数据集
        train_set, val_set, test_set = self.split_dataset(annotations)
        print(f"训练集: {len(train_set)} 张, 验证集: {len(val_set)} 张, 测试集: {len(test_set)} 张")
        
        # 保存YOLO格式
        self.save_yolo_format(train_set, "train")
        self.save_yolo_format(val_set, "val")
        self.save_yolo_format(test_set, "test")
        
        # 创建数据配置文件
        self.create_data_yaml()
        
        print("数据集转换完成!")

if __name__ == "__main__":
    converter = BDFreshFishToYOLO()
    converter.convert()