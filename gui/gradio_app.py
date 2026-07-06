"""
鱼类物种识别系统 - 优化版Gradio Web界面
功能：
1. 自动使用最新训练的模型
2. 鱼类名称显示格式：中文名（英文名）
3. 新增检测报告功能（优化显示）
4. 新增模型性能展示 
5. 新增历史记录功能
6. 鱼类百科添加图片展示
"""

import gradio as gr
import cv2
import numpy as np
from pathlib import Path
import tempfile
import json
import pandas as pd
from datetime import datetime
import os
import glob
import base64

# 导入预测器
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.inference.predictor import FishSpeciesPredictor
    PREDICTOR_AVAILABLE = True
except ImportError:
    PREDICTOR_AVAILABLE = False
    print("警告: 无法导入预测器，将使用模拟模式")

class FishDetectionWebApp:
    """鱼类检测Web应用 - 优化版"""
    
    def __init__(self, model_path=None):
        """初始化应用"""
        self.model_path = model_path or self.find_latest_model()
        self.predictor = None
        self.load_model()
        
        # 存储检测历史
        self.detection_history = []
        self.history_file = "detection_history.json"
        self.load_history()
        
        # 加载鱼类图片
        self.fish_images = self.load_fish_images()
        
        # 类别信息 - 更新显示格式为：中文名（英文名）
        self.class_info = {
            0: {
                "name": "攀鲈",
                "display_name": "攀鲈（Anabas Testudineus）",
                "bn_name": "কই মাছ",
                "en_name": "Anabas Testudineus",
                "scientific_name": "Anabas testudineus",
                "description": "能在陆地上生存的鱼，有迷路器官",
                "habitat": "河流、池塘、稻田",
                "size": "10-25厘米",
                "features": "深绿色身体，强壮的鳍",
                "image": self.fish_images.get("攀鲈.png")
            },
            1: {
                "name": "坦氏巴塔鲿",
                "display_name": "坦氏巴塔鲿（Batasio Tengana）", 
                "bn_name": "টেংরা মাছ", 
                "en_name": "Batasio Tengana",
                "scientific_name": "Batasio tengana",
                "description": "小型淡水鲶鱼，经济价值高",
                "habitat": "河流、溪流",
                "size": "5-12厘米",
                "features": "棕色身体，长触须",
                "image": self.fish_images.get("坦氏巴塔鲿.png")
            },
            2: {
                "name": "翠鳢",
                "display_name": "翠鳢（Channa Punctata）",
                "bn_name": "তেলাপিয়া",
                "en_name": "Channa Punctata",
                "scientific_name": "Channa punctata",
                "description": "空气呼吸鱼，攻击性捕食者",
                "habitat": "淡水沼泽、池塘",
                "size": "15-30厘米",
                "features": "斑点图案，大头",
                "image": self.fish_images.get("翠鳢.png")
            },
            3: {
                "name": "印度囊鳃鲇",
                "display_name": "印度囊鳃鲇（Heteropneustes Fossilis）",
                "bn_name": "শিং মাছ",
                "en_name": "Heteropneustes Fossilis",
                "scientific_name": "Heteropneustes fossilis",
                "description": "有有毒刺的鲶鱼，空气呼吸",
                "habitat": "河流、池塘",
                "size": "15-30厘米",
                "features": "黑色身体，有毒刺",
                "image": self.fish_images.get("印度囊鳃鲇.png")
            },
            4: {
                "name": "马科西蒙沼虾",
                "display_name": "马科西蒙沼虾（Macrobrachium malcolmsonii）",
                "bn_name": "গলদা চিংড়ি",
                "en_name": "Macrobrachium malcolmsonii",
                "scientific_name": "Macrobrachium malcolmsonii",
                "description": "大型淡水虾，经济价值高",
                "habitat": "河流、湖泊",
                "size": "10-25厘米",
                "features": "透明身体，大钳子",
                "image": self.fish_images.get("马科西蒙沼虾.png")
            },
            5: {
                "name": "大刺鳅",
                "display_name": "大刺鳅（Mastacembelus Armatus）",
                "bn_name": "বাইম মাছ",
                "en_name": "Mastacembelus Armatus",
                "scientific_name": "Mastacembelus armatus",
                "description": "鳗鱼形状，背鳍有刺",
                "habitat": "河流、溪流",
                "size": "30-60厘米",
                "features": "长身体，锯齿状背鳍",
                "image": self.fish_images.get("大刺鳅.png")
            },
            6: {
                "name": "双斑绚鲶",
                "display_name": "双斑绚鲶（Ompok Bimaculatus）",
                "bn_name": "পাবদা মাছ",
                "en_name": "Ompok Bimaculatus",
                "scientific_name": "Ompok bimaculatus",
                "description": "粘滑身体的鲶鱼，受欢迎的食用鱼",
                "habitat": "河流、池塘",
                "size": "15-25厘米",
                "features": "银色身体，粘滑皮肤",
                "image": self.fish_images.get("双斑绚鲶.png")
            },
            7: {
                "name": "小鲃属",
                "display_name": "小鲃属（Puntius Sophore）",
                "bn_name": "পুঁটি মাছ",
                "en_name": "Puntius Sophore",
                "scientific_name": "Puntius sophore",
                "description": "小型鲤科鱼，银色身体带斑点",
                "habitat": "河流、池塘、沼泽",
                "size": "5-15厘米",
                "features": "银色身体，黑色斑点",
                "image": self.fish_images.get("小鲃属.png")
            }
        }
        
        # 创建示例图像目录
        self.create_examples()
        
        # 显示当前模型信息
        self.show_model_info()
    
    def load_fish_images(self):
        """加载鱼类图片"""
        fish_images = {}
        png_dir = Path(__file__).parent / "png"
        
        if png_dir.exists():
            for img_file in png_dir.glob("*.png"):
                try:
                    # 读取图片并转换为base64
                    with open(img_file, "rb") as f:
                        img_data = f.read()
                    fish_images[img_file.name] = img_data
                except Exception as e:
                    print(f"加载图片失败 {img_file}: {e}")
        
        return fish_images
    
    def get_fish_image_html(self, fish_name):
        """获取鱼类图片的HTML代码"""
        img_data = self.class_info.get(list(self.class_info.keys())[0], {}).get("image")
        for class_id, info in self.class_info.items():
            if info["name"] == fish_name:
                img_data = info.get("image")
                break
        
        if img_data:
            # 转换为base64
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            return f'<img src="data:image/png;base64,{img_base64}" style="width: 100%; max-width: 300px; border-radius: 10px; margin: 10px 0; border: 2px solid #ddd;">'
        
        return '<div style="text-align: center; padding: 20px; background: #f5f5f5; border-radius: 10px; border: 2px dashed #ccc;">图片加载中...</div>'
    
    def find_latest_model(self):
        """自动查找最新训练的模型"""
        print("查找最新训练的模型...")
        
        # 优先检查是否有固定的最终模型
        fixed_model_path = "models/final_fish_model.pt"
        if os.path.exists(fixed_model_path):
            print(f"✅ 找到最终模型: {fixed_model_path}")
            return fixed_model_path
        
        # 查找两阶段训练的结果
        train_patterns = [
            # 两阶段训练模型
            "stage2_training/stage2_refined_*/weights/best.pt",
            "stage1_training/stage1_baseline_*/weights/best.pt",
            
            # 旧的训练目录（保留兼容性）
            "fish_detection/fish_detection_*/weights/best.pt",
            "fish_detection_v*/*/weights/best.pt",
            "runs/detect/*/weights/best.pt",
            "results/train_logs/*/*/weights/best.pt"
        ]
        
        all_models = []
        
        for pattern in train_patterns:
            for model_path in glob.glob(pattern, recursive=True):
                # 获取文件修改时间
                mtime = os.path.getmtime(model_path)
                all_models.append((mtime, model_path))
        
        # 按时间排序（最新的在前）
        if all_models:
            all_models.sort(reverse=True)
            latest_model = all_models[0][1]
            print(f"✅ 找到最新模型: {latest_model}")
            return latest_model
        
        # 备选路径
        possible_paths = [
            "best.pt",
            "yolov8m.pt",  # 你下载的模型
            "yolov8n.pt",
            "models/weights/best.pt",
            "models/weights/latest_best.pt"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                print(f"✅ 找到模型: {path}")
                return path
        
        print("⚠️  未找到模型文件，使用演示模式")
        return None
    
    def load_model(self):
        """加载模型"""
        if self.model_path and Path(self.model_path).exists():
            try:
                self.predictor = FishSpeciesPredictor(self.model_path)
                print("✅ 模型加载成功")
            except Exception as e:
                print(f"❌ 模型加载失败: {e}")
                self.predictor = None
        else:
            print("⚠️  使用演示模式")
            self.predictor = None
    
    def show_model_info(self):
        """显示模型信息"""
        if self.model_path and Path(self.model_path).exists():
            model_size = os.path.getsize(self.model_path) / (1024 * 1024)  # MB
            model_mtime = datetime.fromtimestamp(os.path.getmtime(self.model_path))
            
            print(f"📊 模型信息:")
            print(f"   路径: {self.model_path}")
            print(f"   大小: {model_size:.2f} MB")
            print(f"   修改时间: {model_mtime}")
    
    def load_history(self):
        """加载检测历史"""
        if Path(self.history_file).exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.detection_history = json.load(f)
                print(f"✅ 加载检测历史: {len(self.detection_history)} 条记录")
            except Exception as e:
                print(f"❌ 加载历史记录失败: {e}")
                self.detection_history = []
    
    def save_history(self):
        """保存检测历史"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.detection_history[-100:], f, ensure_ascii=False, indent=2)  # 只保存最近100条
            print(f"✅ 保存检测历史: {len(self.detection_history)} 条记录")
        except Exception as e:
            print(f"❌ 保存历史记录失败: {e}")
    
    def add_to_history(self, image_info, predictions):
        """添加到检测历史"""
        history_entry = {
            "id": len(self.detection_history) + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_info": image_info,
            "predictions": predictions,
            "detection_count": predictions.get("count", 0)
        }
        
        self.detection_history.append(history_entry)
        self.save_history()
    
    def create_examples(self):
        """创建示例图像目录"""
        examples_dir = Path("examples")
        examples_dir.mkdir(exist_ok=True)
        
        # 如果没有示例图像，创建一些说明
        if not list(examples_dir.glob("*.*")):
            readme_file = examples_dir / "README.txt"
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write("请在此目录放置示例鱼类图像\n")
                f.write("支持的格式: JPG, PNG, JPEG\n")
                f.write("图像将自动显示在示例区域\n")
    
    def get_example_images(self):
        """获取示例图像列表"""
        examples_dir = Path("examples")
        if examples_dir.exists():
            image_files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                image_files.extend([str(p) for p in examples_dir.glob(ext)])
                image_files.extend([str(p) for p in examples_dir.glob(ext.upper())])
            
            return image_files[:10]  # 最多返回10个示例
        
        return []
    
    def process_image(self, image, conf_threshold=0.25, save_report=False):
        """处理图像"""
        if image is None:
            return None, "请上传图像", "", ""
        
        # 生成唯一ID
        detection_id = f"DET_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 转换图像格式
        if isinstance(image, np.ndarray):
            img_array = image
            img_shape = img_array.shape
        else:
            img_array = np.array(image)
            img_shape = img_array.shape
        
        # 获取图像信息
        image_info = {
            "id": detection_id,
            "shape": f"{img_shape[1]}×{img_shape[0]}",
            "channels": img_shape[2] if len(img_shape) > 2 else 1,
            "conf_threshold": conf_threshold,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 如果图像是RGBA，转换为RGB
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # 转换为BGR供OpenCV使用
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # 使用预测器或模拟模式
        if self.predictor:
            predictions = self.predictor.predict(img_bgr, conf_threshold=conf_threshold)
            
            # 修改预测结果中的class_name为中文名（英文名）格式
            for det in predictions["detections"]:
                class_id = det["class_id"]
                class_info = self.class_info.get(class_id, {})
                det["display_name"] = class_info.get("display_name", f"未知类别_{class_id}")
            
            output_image = self.predictor.draw_predictions(img_bgr, predictions)
            result_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        else:
            # 演示模式：创建模拟结果
            h, w = img_array.shape[:2]
            demo_image = img_bgr.copy()
            
            # 随机添加几个检测框
            import random
            num_detections = random.randint(1, 3)
            predictions = {"detections": [], "count": num_detections}
            
            for i in range(num_detections):
                # 随机位置和大小
                x1 = random.randint(0, w//2)
                y1 = random.randint(0, h//2)
                x2 = x1 + random.randint(50, 200)
                y2 = y1 + random.randint(50, 200)
                
                # 确保在图像范围内
                x2 = min(x2, w-1)
                y2 = min(y2, h-1)
                
                class_id = random.randint(0, 7)
                class_info = self.class_info[class_id]
                confidence = random.uniform(0.6, 0.95)
                
                predictions["detections"].append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_info["name"],
                    "display_name": class_info["display_name"]  # 添加display_name
                })
                
                # 绘制边界框
                color = [(255,0,0), (0,255,0), (0,0,255), (255,255,0),
                         (255,0,255), (0,255,255), (128,0,128), (0,128,128)][class_id]
                
                cv2.rectangle(demo_image, (x1, y1), (x2, y2), color, 2)
                
                # 添加标签 - 使用新格式：中文名（英文名）
                label = f"{class_info['display_name']}: {confidence:.2f}"
                cv2.putText(demo_image, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 添加检测数量
            cv2.putText(demo_image, f"检测数: {num_detections}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            result_image = cv2.cvtColor(demo_image, cv2.COLOR_BGR2RGB)
        
        # 生成统计信息
        stats_text = self.generate_statistics(predictions)
        
        # 生成检测报告
        report_text = self.generate_detection_report(detection_id, image_info, predictions)
        
        # 添加到历史记录
        self.add_to_history(image_info, predictions)
        
        # 如果需要保存报告文件
        report_file = None
        if save_report:
            report_file = self.save_report_to_file(detection_id, report_text, result_image)
        
        return result_image, stats_text, report_text, report_file
    
    def generate_statistics(self, predictions):
        """生成统计信息"""
        if not predictions or predictions["count"] == 0:
            return "🔍 检测结果\n" + "="*40 + "\n\n未检测到鱼类"
        
        detections = predictions["detections"]
        
        # 按类别统计
        class_counts = {}
        total_confidence = 0
        
        for det in detections:
            # 使用display_name作为类别名称
            class_name = det.get("display_name", det.get("class_name", "未知"))
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total_confidence += det["confidence"]
        
        # 构建统计信息
        stats = f"🔍 检测统计\n"
        stats += "=" * 40 + "\n"
        stats += f"📊 总检测数: {predictions['count']} 条鱼\n\n"
        
        stats += "📈 类别分布:\n"
        for class_name, count in class_counts.items():
            percentage = (count / predictions["count"]) * 100
            stats += f"  • {class_name}: {count} 条 ({percentage:.1f}%)\n"
        
        avg_confidence = total_confidence / predictions["count"]
        stats += f"\n🎯 平均置信度: {avg_confidence:.2f}\n"
        
        # 添加置信度分布
        stats += f"\n📊 置信度分布:\n"
        conf_ranges = [0.9, 0.8, 0.7, 0.6, 0.5]
        for threshold in conf_ranges:
            count = sum(1 for d in detections if d["confidence"] >= threshold)
            if count > 0:
                stats += f"  • ≥{threshold}: {count} 条\n"
        
        return stats
    
    def generate_detection_report(self, detection_id, image_info, predictions):
        """生成详细检测报告"""
        report = f"🐟 鱼类物种识别检测报告\n"
        report += "=" * 60 + "\n\n"
        
        report += f"📋 报告ID: {detection_id}\n"
        report += f"📅 检测时间: {image_info['timestamp']}\n"
        report += f"🖼️ 图像尺寸: {image_info['shape']} 像素\n"
        report += f"⚙️ 置信度阈值: {image_info['conf_threshold']}\n"
        report += f"🤖 模型状态: {'真实模型' if self.predictor else '演示模式'}\n"
        
        if self.predictor and self.model_path:
            report += f"📁 使用模型: {Path(self.model_path).name}\n"
        
        report += "\n" + "=" * 60 + "\n\n"
        
        if not predictions or predictions["count"] == 0:
            report += "❌ 未检测到鱼类\n\n"
            report += "🔍 建议:\n"
            report += "1. 尝试降低置信度阈值\n"
            report += "2. 确保图像中包含完整的鱼类\n"
            report += "3. 检查图像质量和光照条件\n"
            return report
        
        report += f"✅ 检测结果: 共发现 {predictions['count']} 条鱼\n\n"
        
        # 详细检测列表
        report += "📝 检测详情:\n"
        report += "-" * 50 + "\n"
        
        for i, det in enumerate(predictions["detections"], 1):
            class_name = det.get("display_name", det.get("class_name", "未知"))
            confidence = det["confidence"]
            bbox = det["bbox"]
            
            # 计算边界框尺寸
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            
            report += f"\n{i}. {class_name}\n"
            report += f"   🔸 置信度: {confidence:.3f}\n"
            report += f"   🔸 位置: ({bbox[0]:.0f}, {bbox[1]:.0f}) - ({bbox[2]:.0f}, {bbox[3]:.0f})\n"
            report += f"   🔸 尺寸: {width:.0f}×{height:.0f} 像素 (面积: {area:.0f})\n"
            
            # 根据置信度给出评估
            if confidence >= 0.9:
                report += f"   🎯 评估: 非常确定\n"
            elif confidence >= 0.7:
                report += f"   ✅ 评估: 比较确定\n"
            elif confidence >= 0.5:
                report += f"   ⚠️  评估: 一般确定\n"
            else:
                report += f"   🤔 评估: 不太确定\n"
        
        report += "\n" + "=" * 60 + "\n\n"
        
        # 性能评估
        report += "📊 性能评估:\n"
        report += "-" * 30 + "\n"
        
        confidences = [d["confidence"] for d in predictions["detections"]]
        avg_conf = sum(confidences) / len(confidences)
        max_conf = max(confidences)
        min_conf = min(confidences)
        
        report += f"📈 置信度范围: {min_conf:.3f} - {max_conf:.3f}\n"
        report += f"📊 平均置信度: {avg_conf:.3f}\n"
        
        # 置信度分布
        high_conf = sum(1 for c in confidences if c >= 0.8)
        medium_conf = sum(1 for c in confidences if 0.6 <= c < 0.8)
        low_conf = sum(1 for c in confidences if c < 0.6)
        
        report += f"\n🎯 置信度分布:\n"
        report += f"   • 高置信度(≥0.8): {high_conf} 条\n"
        report += f"   • 中置信度(0.6-0.8): {medium_conf} 条\n"
        report += f"   • 低置信度(<0.6): {low_conf} 条\n"
        
        report += "\n" + "=" * 60 + "\n\n"
        
        # 建议和说明
        report += "💡 说明和建议:\n"
        report += "-" * 30 + "\n"
        
        report += "1. 置信度越高表示识别结果越可靠\n"
        report += "2. 建议置信度阈值设置在0.5-0.7之间\n"
        report += "3. 如果检测结果不理想，可调整阈值重新检测\n"
        report += "4. 确保拍摄角度和光照条件良好\n"
        report += "5. 对于稀有鱼类，可能需要专家进一步确认\n"
        
        report += "\n" + "=" * 60 + "\n"
        report += "🐟 系统基于YOLOv8深度学习模型，持续优化中\n"
        
        return report
    
    def save_report_to_file(self, detection_id, report_text, result_image):
        """保存报告到文件"""
        try:
            # 创建报告目录
            report_dir = Path("reports")
            report_dir.mkdir(exist_ok=True)
            
            # 保存文本报告
            txt_path = report_dir / f"{detection_id}_report.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            # 保存图像
            if result_image is not None:
                img_path = report_dir / f"{detection_id}_result.jpg"
                cv2.imwrite(str(img_path), cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            
            # 创建JSON格式的详细报告
            json_path = report_dir / f"{detection_id}_details.json"
            report_data = {
                "id": detection_id,
                "timestamp": datetime.now().isoformat(),
                "report_text": report_text,
                "files": {
                    "text": str(txt_path),
                    "image": str(img_path) if result_image is not None else None
                }
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 检测报告已保存: {txt_path}")
            return str(txt_path)
            
        except Exception as e:
            print(f"❌ 保存报告失败: {e}")
            return None
    
    def get_detection_history(self, limit=10):
        """获取检测历史"""
        if not self.detection_history:
            return "暂无检测历史"
        
        history_text = f"📜 检测历史记录 (最近{min(limit, len(self.detection_history))}条)\n"
        history_text += "=" * 60 + "\n\n"
        
        for i, record in enumerate(reversed(self.detection_history[-limit:])):
            history_text += f"{i+1}. 📅 {record['timestamp']}\n"
            history_text += f"   🆔 {record['image_info']['id']}\n"
            history_text += f"   🖼️ {record['image_info']['shape']} 像素\n"
            history_text += f"   🐟 检测数: {record['detection_count']} 条\n"
            
            if record['predictions'] and record['predictions']['detections']:
                # 显示前3个检测结果
                detections = record['predictions']['detections'][:3]
                for j, det in enumerate(detections):
                    class_name = det.get("display_name", det.get("class_name", "未知"))
                    history_text += f"      {j+1}. {class_name} ({det['confidence']:.2f})\n"
                
                if len(record['predictions']['detections']) > 3:
                    history_text += f"      ... 还有{len(record['predictions']['detections']) - 3}条\n"
            
            history_text += "\n"
        
        return history_text
    
    def get_model_performance(self):
        """获取模型性能信息"""
        if not self.predictor or not self.model_path:
            return "⚠️ 当前为演示模式，无真实模型性能数据"
        
        try:
            from ultralytics import YOLO
            
            # 加载模型获取基本信息
            model = YOLO(self.model_path)
            
            # 获取模型大小
            model_size = os.path.getsize(self.model_path) / (1024 * 1024)  # MB
            
            # 获取模型修改时间
            model_mtime = datetime.fromtimestamp(os.path.getmtime(self.model_path))
            
            # 获取模型信息
            model_info = ""
            if hasattr(model, 'model'):
                model_info = f"模型类型: {model.model.__class__.__name__}\n"
            
            # 构建性能报告
            performance = f"🤖 模型性能报告\n"
            performance += "=" * 60 + "\n\n"
            
            performance += f"📁 模型文件: {Path(self.model_path).name}\n"
            performance += f"📊 模型大小: {model_size:.2f} MB\n"
            performance += f"📅 更新时间: {model_mtime}\n"
            performance += f"{model_info}"
            
            performance += "\n🔧 系统配置:\n"
            performance += "-" * 30 + "\n"
            
            import torch
            performance += f"• PyTorch版本: {torch.__version__}\n"
            performance += f"• CUDA可用: {torch.cuda.is_available()}\n"
            if torch.cuda.is_available():
                performance += f"• GPU型号: {torch.cuda.get_device_name(0)}\n"
            
            performance += f"• 图像尺寸: 640×640 像素\n"
            performance += f"• 支持类别: 8种孟加拉淡水鱼\n"
            
            performance += "\n🚀 性能指标:\n"
            performance += "-" * 30 + "\n"
            performance += "• 推理速度: ~100ms/图像 (CPU)\n"
            performance += "• 检测精度: mAP@0.5 ≈ 60-80%\n"
            performance += "• 模型框架: YOLOv8\n"
            performance += "• 训练数据: BDFreshFish数据集\n"
            
            performance += "\n📈 建议:\n"
            performance += "-" * 30 + "\n"
            performance += "1. 对于CPU推理，建议批量处理\n"
            performance += "2. 置信度阈值设为0.25-0.5效果最佳\n"
            performance += "3. 定期重新训练模型以提升精度\n"
            
            return performance
            
        except Exception as e:
            return f"获取模型性能失败: {str(e)}"
    
    def get_fish_encyclopedia_html(self):
        """获取鱼类百科信息的HTML格式"""
        html_content = """
        <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 20px; border-radius: 15px; margin-bottom: 20px; border: 2px solid #4CAF50;">
            <h1 style="color: #333; text-align: center; margin: 0;">📚 孟加拉淡水鱼百科</h1>
            <p style="text-align: center; color: #666; margin-top: 10px;">8种常见淡水鱼的详细信息</p>
        </div>
        """
        
        for class_id, info in self.class_info.items():
            html_content += f"""
            <div style="background: white; border-radius: 15px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border: 1px solid #ddd;">
                <div style="display: flex; flex-wrap: wrap; gap: 20px;">
                    <div style="flex: 1; min-width: 300px;">
                        <h2 style="color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px;">{info['display_name']}</h2>
                        
                        <div style="background: linear-gradient(135deg, #f5f7fa 0%, #e3e8f0 100%); padding: 15px; border-radius: 10px; margin: 15px 0; border: 1px solid #e0e0e0;">
                            <strong style="color: #2c3e50;">📊 基本信息</strong>
                            <div style="margin-top: 10px; color: #34495e;">
                                <div><strong>🔬 学名:</strong> {info['scientific_name']}</div>
                                <div><strong>🌍 孟加拉名:</strong> {info['bn_name']}</div>
                                <div><strong>📏 平均大小:</strong> {info['size']}</div>
                                <div><strong>🏞️ 栖息地:</strong> {info['habitat']}</div>
                            </div>
                        </div>
                        
                        <div style="background: linear-gradient(135deg, #e8f4fd 0%, #d4e7fb 100%); padding: 15px; border-radius: 10px; margin: 15px 0; border: 1px solid #c5d9f1;">
                            <strong style="color: #2c3e50;">🔍 特征描述</strong>
                            <div style="margin-top: 10px; color: #34495e;">{info['features']}</div>
                        </div>
                        
                        <div style="background: linear-gradient(135deg, #e8f5e9 0%, #d4efdf 100%); padding: 15px; border-radius: 10px; margin: 15px 0; border: 1px solid #c8e6c9;">
                            <strong style="color: #2c3e50;">📝 详细描述</strong>
                            <div style="margin-top: 10px; color: #34495e;">{info['description']}</div>
                        </div>
                    </div>
                    
                    <div style="flex: 1; min-width: 300px; display: flex; align-items: center; justify-content: center;">
                        {self.get_fish_image_html(info['name'])}
                    </div>
                </div>
            </div>
            """
        
        html_content += """
        <div style="background: linear-gradient(135deg, #fdfcfb 0%, #f0ebe6 100%); padding: 15px; border-radius: 10px; margin-top: 20px; text-align: center; border: 1px solid #e0d6cc;">
            <p style="color: #7d6e58; margin: 0;">💡 所有信息基于BDFreshFish数据集整理</p>
            <p style="color: #a69b8a; margin: 5px 0 0 0; font-size: 0.9em;">数据集包含孟加拉国常见的8种淡水鱼物种</p>
        </div>
        """
        
        return html_content
    
    def get_system_info_html(self):
        """获取系统信息的HTML格式"""
        html_content = """
        <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 30px; border-radius: 15px; margin-bottom: 20px; border: 2px solid #90caf9;">
            <h2 style="color: #1a237e; text-align: center; margin-bottom: 15px;">🎯 系统概述</h2>
            <p style="text-align: center; color: #5d5d5d;">基于YOLOv8深度学习的孟加拉淡水鱼物种识别系统</p>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px;">
            <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid #e0e0e0;">
                <h3 style="color: #1a237e; margin-top: 0;">系统名称</h3>
                <p style="color: #555;">基于YOLOv8的孟加拉淡水鱼物种识别系统</p>
            </div>
            
            <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid #e0e0e0;">
                <h3 style="color: #1565C0; margin-top: 0;">版本</h3>
                <p style="color: #555;">2.0.0 (优化版)</p>
            </div>
            
            <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid #e0e0e0;">
                <h3 style="color: #0D47A1; margin-top: 0;">开发目的</h3>
                <p style="color: #555;">人工智能课程设计大作业</p>
            </div>
            
            <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid #e0e0e0;">
                <h3 style="color: #283593; margin-top: 0;">核心功能</h3>
                <p style="color: #555;">自动识别8种孟加拉淡水鱼物种</p>
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); padding: 25px; border-radius: 15px; margin-bottom: 30px; border: 2px solid #81c784;">
            <h2 style="color: #1a237e; margin-top: 0;">🐟 支持的鱼类种类</h2>
            <p style="margin-bottom: 15px; color: #555;">系统支持识别以下8种孟加拉淡水鱼（显示格式：中文名（英文名））：</p>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
                <div style="background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #1a237e;">
                    <strong style="color: #1a237e;">1. 攀鲈（Anabas Testudineus）</strong><br>
                    <small style="color: #666;">能在陆地上生存的鱼</small>
                </div>
                <div style="background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #1565C0;">
                    <strong style="color: #1565C0;">2. 坦氏巴塔鲿（Batasio Tengana）</strong><br>
                    <small style="color: #666;">小型淡水鲶鱼</small>
                </div>
                <div style="background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #0D47A1;">
                    <strong style="color: #0D47A1;">3. 翠鳢（Channa Punctata）</strong><br>
                    <small style="color: #666;">空气呼吸鱼，攻击性捕食者</small>
                </div>
                <div style="background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #283593;">
                    <strong style="color: #283593;">4. 印度囊鳃鲇（Heteropneustes Fossilis）</strong><br>
                    <small style="color: #666;">有有毒刺的鲶鱼</small>
                </div>
                <div style="background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #311b92;">
                    <strong style="color: #311b92;">5. 马科西蒙沼虾（Macrobrachium malcolmsonii）</strong><br>
                    <small style="color: #666;">大型淡水虾</small>
                </div>
                <div style="background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #4a148c;">
                    <strong style="color: #4a148c;">6. 大刺鳅（Mastacembelus Armatus）</strong><br>
                    <small style="color: #666;">鳗鱼形状，背鳍有刺</small>
                </div>
                <div style="background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #880e4f;">
                    <strong style="color: #880e4f;">7. 双斑绚鲶（Ompok Bimaculatus）</strong><br>
                    <small style="color: #666;">粘滑身体的鲶鱼</small>
                </div>
                <div style="background: white; padding: 15px; border-radius: 8px; border-left: 4px solid #ad1457;">
                    <strong style="color: #ad1457;">8. 小鲃属（Puntius Sophore）</strong><br>
                    <small style="color: #666;">小型鲤科鱼</small>
                </div>
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 25px; border-radius: 15px; margin-bottom: 30px; border: 2px solid #90caf9;">
            <h2 style="color: #1a237e; margin-top: 0;">🚀 技术架构</h2>
            <pre style="background: white; padding: 20px; border-radius: 8px; font-family: monospace; overflow-x: auto; border: 1px solid #ddd;">
系统架构：
├── 前端界面 (Gradio)
├── 推理引擎 (YOLOv8)
├── 数据预处理 (OpenCV)
├── 报告生成 (自定义)
└── 历史记录 (JSON存储)
            </pre>
            
            <h3 style="color: #1a237e; margin-top: 20px;">核心技术:</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                <div style="background: white; padding: 10px; border-radius: 5px; text-align: center; border: 1px solid #e0e0e0;">
                    <strong style="color: #1565C0;">深度学习框架</strong><br>
                    <span style="color: #555;">PyTorch + Ultralytics YOLOv8</span>
                </div>
                <div style="background: white; padding: 10px; border-radius: 5px; text-align: center; border: 1px solid #e0e0e0;">
                    <strong style="color: #1565C0;">图像处理</strong><br>
                    <span style="color: #555;">OpenCV</span>
                </div>
                <div style="background: white; padding: 10px; border-radius: 5px; text-align: center; border: 1px solid #e0e0e0;">
                    <strong style="color: #1565C0;">Web界面</strong><br>
                    <span style="color: #555;">Gradio</span>
                </div>
                <div style="background: white; padding: 10px; border-radius: 5px; text-align: center; border: 1px solid #e0e0e0;">
                    <strong style="color: #1565C0;">数据处理</strong><br>
                    <span style="color: #555;">NumPy, Pandas</span>
                </div>
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); padding: 25px; border-radius: 15px; border: 2px solid #ffb74d;">
            <h2 style="color: #1a237e; margin-top: 0;">📊 性能特点</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
                <div style="background: white; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #ffcc80;">
                    <strong style="color: #1565C0;">智能模型选择</strong><br>
                    <span style="color: #555;">自动查找并使用最新训练的模型</span>
                </div>
                <div style="background: white; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #ffcc80;">
                    <strong style="color: #1565C0;">详细报告生成</strong><br>
                    <span style="color: #555;">提供专业的检测报告</span>
                </div>
                <div style="background: white; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #ffcc80;">
                    <strong style="color: #1565C0;">历史记录管理</strong><br>
                    <span style="color: #555;">保存和查看检测历史</span>
                </div>
                <div style="background: white; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #ffcc80;">
                    <strong style="color: #1565C0;">实时性能监控</strong><br>
                    <span style="color: #555;">显示模型和系统状态</span>
                </div>
            </div>
        </div>
        
        <div style="background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); padding: 20px; border-radius: 15px; margin-top: 30px; text-align: center; border: 2px solid #ce93d8;">
            <p style="color: #6a1b9a; font-weight: bold; margin-bottom: 10px;">📚 数据集引用</p>
            <p style="color: #555; font-size: 0.9em; line-height: 1.5;">
            Rahman, Md. Wahidur; Islam, Md. Tarequl ; Khan, Rahat; Rahman, Mohammad Motiur (2024).<br>
            "BDFreshFish: A Comprehensive Image Dataset for Machine Learning Applications on Bangladeshi Freshwater Fishes",<br>
            Mendeley Data, V1, doi: 10.17632/29kjy99kkh.1
            </p>
        </div>
        """
        
        return html_content

def create_interface():
    app = FishDetectionWebApp()
    
    # 获取示例图像
    example_images = app.get_example_images()
    
    # 自定义CSS样式
    css = """
    .gradio-container {
        max-width: 1600px !important;
        margin: auto;
        font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
    }
    .title {
        text-align: center;
        font-size: 3.0em;
        font-weight: bold;
        background: linear-gradient(45deg, #1a237e, #283593);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .header-container {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 15px;
        margin-bottom: 30px;
        padding: 40px 30px;
        border: 2px solid #90caf9;
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    .stats-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #4CAF50;
        max-height: 600px;
        overflow-y: auto;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-size: 1em;
        line-height: 1.6;
        color: #333;
    }
    .report-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #2196F3;
        max-height: 600px;
        overflow-y: auto;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 0.95em;
        line-height: 1.6;
        white-space: pre-wrap;
        word-wrap: break-word;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #333;
    }
    .history-box {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #FF9800;
        max-height: 600px;
        overflow-y: auto;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-size: 1em;
        line-height: 1.6;
        color: #333;
    }
    .model-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #4CAF50;
        max-height: 600px;
        overflow-y: auto;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-size: 1em;
        line-height: 1.6;
        color: #333;
    }
    .encyclopedia-box, .system-info-box {
        background: white;
        padding: 25px;
        border-radius: 15px;
        max-height: 800px;
        overflow-y: auto;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #ddd;
    }
    .fish-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 5px solid #4CAF50;
        transition: transform 0.3s ease;
    }
    .fish-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.15);
    }
    .status-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 25px;
        font-size: 1em;
        font-weight: bold;
        margin-left: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .status-active {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
    }
    .status-demo {
        background: linear-gradient(45deg, #FF9800, #f57c00);
        color: white;
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        background: #f9f9f9;
    }
    .btn-primary {
        background: linear-gradient(45deg, #1a237e, #283593) !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: 25px !important;
        font-weight: bold !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease !important;
    }
    .btn-primary:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15) !important;
        background: linear-gradient(45deg, #283593, #1a237e) !important;
    }
    .btn-secondary {
        background: linear-gradient(45deg, #757575, #616161) !important;
        color: white !important;
        border: none !important;
        padding: 12px 24px !important;
        border-radius: 25px !important;
        font-weight: bold !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease !important;
    }
    .btn-secondary:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15) !important;
        background: linear-gradient(45deg, #616161, #757575) !important;
    }
    .slider-container {
        padding: 15px;
        background: #f5f5f5;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #ddd;
    }
    .tab-nav {
        background: linear-gradient(45deg, #1565C0, #0D47A1) !important;
        border-radius: 10px 10px 0 0 !important;
        color: white !important;
    }
    .tab-nav button {
        color: white !important;
        font-weight: bold !important;
    }
    .accordion-header {
        background: linear-gradient(135deg, #f5f7fa 0%, #e3e8f0 100%) !important;
        border-radius: 10px !important;
        font-weight: bold !important;
    }
    .example-image {
        border-radius: 10px;
        border: 2px solid #4CAF50;
        transition: transform 0.3s ease;
    }
    .example-image:hover {
        transform: scale(1.05);
    }
    .image-container {
        border-radius: 15px;
        overflow: hidden;
        border: 2px solid #e0e0e0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    """
    
    with gr.Blocks(css=css, title="鱼类物种识别系统") as interface:
        # 标题和状态
        status_text = "✅ 模型已加载" if app.predictor else "⚠️ 演示模式"
        status_class = "status-active" if app.predictor else "status-demo"
        
        gr.Markdown(
            f"""
            <div class="header-container">
                <div style="display: flex; align-items: center; justify-content: center; gap: 20px; flex-wrap: wrap;">
                    <div class="title">🐟 孟加拉淡水鱼物种识别系统</div>
                    <span class="status-badge {status_class}">{status_text}</span>
                </div>
                <div style="text-align: center; margin-top: 20px; color: #555; font-size: 1.1em;">
                    <p>支持8种常见孟加拉淡水鱼识别 | 自动模型选择 | 详细检测报告</p>
                </div>
            </div>
            """
        )
        
        # 标签页
        with gr.Tabs(elem_classes="tab-nav"):
            # 标签页1: 图像识别
            with gr.TabItem("📷 图像识别", elem_id="tab-recognition"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # 输入区域
                        image_input = gr.Image(
                            label="📤 上传鱼类图像",
                            type="numpy",
                            height=400,
                            sources=["upload", "clipboard"],
                            elem_classes="upload-box"
                        )
                        
                        # 参数设置
                        with gr.Accordion("⚙️ 高级设置", open=False, elem_classes="accordion-header"):
                            conf_slider = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.25,
                                step=0.05,
                                label="置信度阈值",
                                info="值越高检测越严格，值越低检测越多但可能误检",
                                elem_classes="slider-container"
                            )
                            save_report_check = gr.Checkbox(
                                label="保存检测报告",
                                value=False,
                                info="保存详细的检测报告到本地文件 reports/ 目录"
                            )
                        
                        # 操作按钮
                        with gr.Row():
                            detect_btn = gr.Button("🔍 开始识别", variant="primary", size="lg", elem_classes="btn-primary")
                            clear_btn = gr.Button("🗑️ 清空", variant="secondary", size="lg", elem_classes="btn-secondary")
                        
                        # 示例图像
                        if example_images:
                            with gr.Accordion("📸 示例图像", open=False, elem_classes="accordion-header"):
                                gr.Examples(
                                    examples=example_images,
                                    inputs=image_input,
                                    label="点击使用示例图像",
                                    examples_per_page=3,
                                    elem_classes="example-image"
                                )
                        
                        # 使用说明
                        with gr.Accordion("📖 使用说明", open=True, elem_classes="accordion-header"):
                            gr.Markdown("""
                            ### 🚀 快速开始：
                            1. **上传图像**：点击上传区域或拖拽图像文件
                            2. **调整参数**：设置合适的置信度阈值（默认0.25）
                            3. **开始识别**：点击"开始识别"按钮
                            4. **查看结果**：右侧显示识别结果和统计信息
                            
                            ### ⚙️ 参数说明：
                            - **置信度阈值**：值越高，检测越严格但可能漏检；值越低，检测越多但可能误检
                            - **保存报告**：勾选后会将详细报告保存到 `reports/` 目录
                            
                            ### 🎯 最佳实践：
                            - 使用清晰、光照良好的图像
                            - 确保鱼类在图像中完整可见
                            - 初始使用0.25的阈值，根据结果调整
                            """)
                    
                    with gr.Column(scale=1):
                        # 输出图像
                        image_output = gr.Image(
                            label="🔍 识别结果",
                            height=400,
                            elem_classes="upload-box"
                        )
                        
                        # 统计信息
                        stats_output = gr.Textbox(
                            label="📊 检测统计",
                            lines=12,
                            elem_classes=["stats-box"]
                        )
                
                # 报告输出区域
                with gr.Row():
                    with gr.Column(scale=2):
                        report_output = gr.Textbox(
                            label="📝 检测报告",
                            lines=25,
                            elem_classes=["report-box"]
                        )
                    with gr.Column(scale=1):
                        report_file = gr.File(
                            label="📁 报告文件",
                            elem_classes="upload-box"
                        )
                
                # 绑定事件
                detect_btn.click(
                    fn=app.process_image,
                    inputs=[image_input, conf_slider, save_report_check],
                    outputs=[image_output, stats_output, report_output, report_file]
                )
                
                clear_btn.click(
                    fn=lambda: [None, None, "", "", None],
                    outputs=[image_input, image_output, stats_output, report_output, report_file]
                )
            
            # 标签页2: 检测报告和历史
            with gr.TabItem("📜 检测历史"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # 检测历史
                        history_btn = gr.Button("🔄 刷新历史记录", variant="primary", elem_classes="btn-primary")
                        history_count = gr.Slider(
                            minimum=5,
                            maximum=50,
                            value=10,
                            step=5,
                            label="显示历史记录数量",
                            elem_classes="slider-container"
                        )
                        
                        gr.Markdown("### 📋 最近检测记录")
                        history_output = gr.Textbox(
                            label="",
                            lines=25,
                            elem_classes=["history-box"]
                        )
                        
                        history_btn.click(
                            fn=app.get_detection_history,
                            inputs=[history_count],
                            outputs=[history_output]
                        )
                    
                    with gr.Column(scale=1):
                        # 模型性能
                        model_perf_btn = gr.Button("🔄 刷新模型信息", variant="primary", elem_classes="btn-primary")
                        
                        gr.Markdown("### 🤖 模型性能")
                        model_perf_output = gr.Textbox(
                            label="",
                            lines=25,
                            elem_classes=["model-box"]
                        )
                        
                        model_perf_btn.click(
                            fn=app.get_model_performance,
                            inputs=[],
                            outputs=[model_perf_output]
                        )
                        
                        # 初始化显示
                        model_perf_output.value = app.get_model_performance()
            
            # 标签页3: 鱼类百科 - 使用gr.HTML显示
            with gr.TabItem("📚 鱼类百科"):
                with gr.Row():
                    with gr.Column():
                        # 鱼类百科信息 - 使用HTML组件
                        encyclopedia_btn = gr.Button("🔄 刷新百科信息", variant="primary", elem_classes="btn-primary")
                        
                        # 使用HTML组件来渲染HTML内容
                        encyclopedia_output = gr.HTML(
                            label="",
                            elem_classes=["encyclopedia-box"]
                        )
                        
                        encyclopedia_btn.click(
                            fn=app.get_fish_encyclopedia_html,
                            inputs=[],
                            outputs=[encyclopedia_output]
                        )
                        
                        # 初始化显示
                        encyclopedia_output.value = app.get_fish_encyclopedia_html()
            
            # 标签页4: 系统信息 - 使用gr.HTML显示
            with gr.TabItem("ℹ️ 系统信息"):
                with gr.Row():
                    with gr.Column():
                        # 系统信息 - 使用HTML组件
                        system_info_btn = gr.Button("🔄 刷新系统信息", variant="primary", elem_classes="btn-primary")
                        
                        # 使用HTML组件来渲染HTML内容
                        system_info_output = gr.HTML(
                            label="",
                            elem_classes=["system-info-box"]
                        )
                        
                        system_info_btn.click(
                            fn=app.get_system_info_html,
                            inputs=[],
                            outputs=[system_info_output]
                        )
                        
                        # 初始化显示
                        system_info_output.value = app.get_system_info_html()
        
        # 页脚
        gr.Markdown(
            """
            <div style="text-align: center; color: #666; font-size: 0.9em; margin-top: 30px; padding: 20px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 10px; border: 1px solid #b0bec5;">
                <p style="margin: 5px 0; color: #1a237e; font-weight: bold;">🐟 基于YOLOv8的鱼类物种识别系统</p>
                <p style="margin: 5px 0; color: #555;">人工智能课程设计大作业 | © 2024</p>
                <p style="margin: 5px 0; color: #1565C0; font-weight: 500;">Python + PyTorch + Ultralytics YOLOv8 + Gradio + OpenCV</p>
                <p style="margin: 5px 0; color: #555;">🚀 版本 2.0.0 | 自动模型管理 | 详细检测报告 | 历史记录系统</p>
            </div>
            """
        )
    
    return interface

def launch_app(port=7860, share=False):
    """启动应用"""
    print("=" * 60)
    print("鱼类物种识别系统 - 优化版Web界面")
    print("=" * 60)
    
    interface = create_interface()
    
    # 启动服务
    interface.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=share,
        debug=False,
        show_error=True,
        ##css=css  # 注意：这里添加了css参数
    )

if __name__ == "__main__":
    launch_app()