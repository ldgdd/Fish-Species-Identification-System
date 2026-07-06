import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Union
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results
import glob
import os

class FishSpeciesPredictor:
    def __init__(self, model_path: str = None, device: str = None):
        """
        初始化预测器
        
        Args:
            model_path: 模型权重路径，如果为None则自动查找最新模型
            device: 推理设备 ('cpu', 'cuda', '0', etc.)
        """
        self.model_path = model_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = None
        self.load_model()
        
    def load_model(self):
        """加载YOLOv8模型"""
        # 如果传入的model_path是None，则自动查找最新模型
        if self.model_path is None:
            self.model_path = self.find_latest_model()
        
        print(f"加载模型: {self.model_path}")
        
        if self.model_path and os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
        else:
            print("⚠️  模型文件不存在，使用预训练模型")
            self.model = YOLO("yolov8n.pt")
        
        # 获取类别名称
        if hasattr(self.model, 'names'):
            self.class_names = self.model.names
            # 确保类别名称是中文显示格式
            self.class_names = self.get_display_names(self.class_names)
        else:
            # 默认类别名称（使用中文显示格式）
            self.class_names = self.get_display_names()
            
        print(f"模型加载完成，设备: {self.device}")
        print(f"检测类别: {self.class_names}")
    
    def find_latest_model(self):
        """查找最新训练的模型"""
        # 检查路径
        check_paths = [
            "models/final_fish_model.pt",
            "stage2_training/stage2_refined_*/weights/best.pt",
            "stage1_training/stage1_baseline_*/weights/best.pt",
            "best.pt",
            "yolov8m.pt",
            "yolov8n.pt"
        ]
        
        for path_pattern in check_paths:
            # 处理通配符
            if '*' in path_pattern:
                matches = glob.glob(path_pattern, recursive=True)
                if matches:
                    # 按修改时间排序，取最新的
                    matches.sort(key=os.path.getmtime, reverse=True)
                    return matches[0]
            else:
                if os.path.exists(path_pattern):
                    return path_pattern
        
        return None

    def get_display_names(self, original_names=None):
        """获取中文显示格式的类别名称"""
        if original_names is None:
            original_names = {}
        
        # 中文显示格式映射
        display_names = {
            0: "攀鲈（Anabas Testudineus）",
            1: "坦氏巴塔鲿（Batasio Tengana）", 
            2: "翠鳢（Channa Punctata）",
            3: "印度囊鳃鲇（Heteropneustes Fossilis）",
            4: "马科西蒙沼虾（Macrobrachium malcolmsonii）",
            5: "大刺鳅（Mastacembelus Armatus）", 
            6: "双斑绚鲶（Ompok Bimaculatus）",
            7: "小鲃属（Puntius Sophore）"
        }
        
        # 如果original_names不为空，确保索引一致
        if original_names:
            result = {}
            for idx, name in original_names.items():
                idx_int = int(idx) if isinstance(idx, str) else idx
                result[idx_int] = display_names.get(idx_int, name)
            return result
        
        return display_names
    
    def preprocess_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """预处理图像"""
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"无法读取图像: {image}")
                
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb
    
    def predict(self, 
                image: Union[str, np.ndarray, List], 
                conf_threshold: float = 0.25,
                iou_threshold: float = 0.45,
                max_detections: int = 100) -> Dict:
        """
        执行预测
        
        Args:
            image: 输入图像或图像列表
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
            max_detections: 最大检测数量
            
        Returns:
            预测结果字典
        """
        # 预处理
        if isinstance(image, list):
            images = [self.preprocess_image(img) for img in image]
        else:
            images = [self.preprocess_image(image)]
        
        # 执行推理
        results = self.model(
            images,
            conf=conf_threshold,
            iou=iou_threshold,
            max_det=max_detections,
            device=self.device
        )
        
        # 处理结果
        predictions = []
        for i, result in enumerate(results):
            if isinstance(result, Results):
                pred_data = self._parse_result(result)
                predictions.append(pred_data)
                
        return predictions if len(predictions) > 1 else predictions[0]
    
    def _parse_result(self, result: Results) -> Dict:
        """解析YOLO结果"""
        if result.boxes is None:
            return {"detections": [], "count": 0}
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        detections = []
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            class_name = self.class_names.get(int(cls_id), f"Class_{cls_id}")
            
            detection = {
                "bbox": box.tolist(),
                "confidence": float(conf),
                "class_id": int(cls_id),
                "class_name": class_name,  # 使用中文显示格式
                "center_x": float((box[0] + box[2]) / 2),
                "center_y": float((box[1] + box[3]) / 2),
                "width": float(box[2] - box[0]),
                "height": float(box[3] - box[1])
            }
            detections.append(detection)
            
        return {
            "detections": detections,
            "count": len(detections),
            "image_shape": result.orig_shape
        }
    
    def draw_predictions(self, 
                         image: np.ndarray, 
                         predictions: Dict, 
                         show_labels: bool = True,
                         show_conf: bool = True) -> np.ndarray:
        """
        在图像上绘制预测结果
        
        Args:
            image: 原始图像
            predictions: 预测结果
            show_labels: 是否显示标签
            show_conf: 是否显示置信度
            
        Returns:
            绘制后的图像
        """
        # 创建图像副本
        output_image = image.copy()
        
        # 颜色映射
        colors = [
            (255, 0, 0),    # 红色
            (0, 255, 0),    # 绿色
            (0, 0, 255),    # 蓝色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 紫色
            (0, 255, 255),  # 黄色
            (128, 0, 128),  # 紫色
            (0, 128, 128)   # 橄榄色
        ]
        
        # 绘制每个检测
        for detection in predictions.get("detections", []):
            bbox = detection["bbox"]
            class_id = detection["class_id"]
            confidence = detection["confidence"]
            class_name = detection["class_name"]
            
            # 获取颜色
            color = colors[class_id % len(colors)]
            
            # 转换为整数坐标
            x1, y1, x2, y2 = map(int, bbox)
            
            # 绘制边界框
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
            
            # 创建标签文本
            label = class_name
            if show_conf:
                label += f" {confidence:.2f}"
                
            if show_labels:
                # 计算标签背景大小
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # 绘制标签背景
                cv2.rectangle(
                    output_image,
                    (x1, y1 - label_height - baseline - 5),
                    (x1 + label_width, y1),
                    color,
                    -1
                )
                
                # 绘制标签文本
                cv2.putText(
                    output_image,
                    label,
                    (x1, y1 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
                
        # 添加统计信息
        total_count = predictions.get("count", 0)
        cv2.putText(
            output_image,
            f"Total Detections: {total_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        
        return output_image
    
    def batch_predict(self, 
                      image_dir: str, 
                      output_dir: str = None,
                      conf_threshold: float = 0.25) -> List[Dict]:
        """
        批量预测目录中的所有图像
        
        Args:
            image_dir: 图像目录路径
            output_dir: 输出目录路径
            conf_threshold: 置信度阈值
            
        Returns:
            所有图像的预测结果列表
        """
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise ValueError(f"目录不存在: {image_dir}")
            
        # 获取所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))
            
        print(f"找到 {len(image_files)} 张图像")
        
        # 创建输出目录
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
        all_predictions = []
        
        # 批量处理
        for img_path in image_files:
            try:
                # 预测
                predictions = self.predict(str(img_path), conf_threshold=conf_threshold)
                
                # 保存结果图像
                if output_dir:
                    image = cv2.imread(str(img_path))
                    output_image = self.draw_predictions(image, predictions)
                    
                    output_path = output_dir / f"pred_{img_path.name}"
                    cv2.imwrite(str(output_path), output_image)
                    
                all_predictions.append({
                    "image_path": str(img_path),
                    "predictions": predictions
                })
                
            except Exception as e:
                print(f"处理图像 {img_path} 时出错: {e}")
                
        return all_predictions

# 使用示例
if __name__ == "__main__":
    # 示例1：不指定模型路径，自动查找最新模型
    predictor = FishSpeciesPredictor(device="cpu")
    
    # 示例2：指定模型路径
    # predictor = FishSpeciesPredictor(
    #     model_path="results/train_logs/fish_detection/weights/best.pt",
    #     device="cpu"
    # )
    
    # 单张图像预测
    result = predictor.predict("test_image.jpg")
    print(f"检测到 {result['count']} 条鱼")
    
    # 可视化结果
    image = cv2.imread("test_image.jpg")
    if image is not None:
        output_image = predictor.draw_predictions(image, result)
        cv2.imwrite("output.jpg", output_image)
        print("结果已保存到 output.jpg")
    else:
        print("测试图像不存在，请准备测试图像")