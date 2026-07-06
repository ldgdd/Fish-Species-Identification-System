import os
import sys
import yaml
from ultralytics import YOLO
from datetime import datetime
import torch

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# 现在可以导入训练配置
try:
    # 尝试绝对导入
    from src.training.train_config import TrainingConfig
except ImportError:
    # 如果失败，尝试从当前目录导入
    from train_config import TrainingConfig

class YOLOv8Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.setup_device()
        
    def setup_device(self):
        """设置训练设备"""
        if self.config.device != "cpu" and torch.cuda.is_available():
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("使用CPU")
            self.config.device = "cpu"
            
    def prepare_model(self):
        """准备YOLOv8模型"""
        if self.config.pretrained and self.config.weights_path:
            print(f"加载预训练权重: {self.config.weights_path}")
            self.model = YOLO(self.config.weights_path)
        elif self.config.pretrained:
            print(f"加载预训练的 {self.config.model_name} 模型")
            self.model = YOLO(f"{self.config.model_name}.pt")
        else:
            print(f"创建新的 {self.config.model_name} 模型")
            self.model = YOLO(f"{self.config.model_name}.yaml")
            
    def train(self):
        """训练模型"""
        print("开始训练YOLOv8模型...")
        
        # 准备模型
        self.prepare_model()
        
        # 训练参数
        train_args = {
            "data": self.config.data_yaml,
            "epochs": self.config.epochs,
            "batch": self.config.batch_size,
            "imgsz": self.config.img_size,
            "workers": self.config.workers,
            "device": self.config.device,
            "project": self.config.project,
            "name": self.config.name,
            "exist_ok": True,
            "save": True,
            "save_period": 10,
            "patience": 50,
            "lr0": self.config.lr0,
            "lrf": self.config.lrf,
            "momentum": self.config.momentum,
            "weight_decay": self.config.weight_decay,
            "hsv_h": self.config.hsv_h,
            "hsv_s": self.config.hsv_s,
            "hsv_v": self.config.hsv_v,
            "flipud": self.config.flipud,
            "fliplr": self.config.fliplr,
            "mosaic": self.config.mosaic,
            "mixup": self.config.mixup,
            "verbose": True
        }
        
        # 开始训练
        results = self.model.train(**train_args)
        
        print("训练完成!")
        return results
    
    def validate(self, model_path=None):
        """验证模型"""
        if model_path is None:
            # 使用最后一次的训练权重
            model_path = os.path.join(self.config.save_dir, self.config.project, 
                                     self.config.name, "weights", "best.pt")
            
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            return None
            
        print(f"验证模型: {model_path}")
        model = YOLO(model_path)
        
        # 在验证集上验证
        metrics = model.val(
            data=self.config.data_yaml,
            batch=self.config.batch_size,
            imgsz=self.config.img_size,
            device=self.config.device
        )
        
        return metrics
    
    def export_model(self, model_path=None, format="onnx"):
        """导出模型为其他格式"""
        if model_path is None:
            model_path = os.path.join(self.config.save_dir, self.config.project,
                                     self.config.name, "weights", "best.pt")
            
        print(f"导出模型: {model_path} -> {format}")
        model = YOLO(model_path)
        
        export_path = model.export(format=format)
        print(f"模型已导出到: {export_path}")
        return export_path

# 修改 main 函数（在文件底部）
def main():
    """主训练函数 - 现在使用两阶段训练"""
    print("=" * 60)
    print("鱼类物种识别系统 - 训练模块")
    print("=" * 60)
    
    # 检查是否有智能边界框数据
    if not os.path.exists("data/smart_annotated/data.yaml"):
        print("⚠️ 未找到智能边界框数据，请先运行:")
        print("   python src/utils/smart_bbox_generator.py")
        
        # 询问是否要生成
        choice = input("是否现在生成智能边界框? (y/n): ")
        if choice.lower() == 'y':
            try:
                from src.utils.smart_bbox_generator import SmartBBoxGenerator
                generator = SmartBBoxGenerator()
                generator.convert_dataset()
            except Exception as e:
                print(f"❌ 生成失败: {e}")
                return
    
    # 运行两阶段训练
    try:
        from two_stage_trainer import TwoStageYOLOTrainer
        trainer = TwoStageYOLOTrainer()
        results = trainer.run_complete_pipeline()
        
        if results:
            print("\n✅ 训练完成!")
            print(f"最终模型: {results.get('stage2_model', '未找到')}")
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()