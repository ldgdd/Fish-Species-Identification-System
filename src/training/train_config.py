import os
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TrainingConfig:
    # 数据配置
    data_yaml: str = "data/processed_v2/data.yaml"
    
    # 模型配置
    model_name: str = "yolov8l"
    pretrained: bool = True
    weights_path: Optional[str] = None
    
    # 训练参数
    epochs: int = 300  # 增加训练轮数
    batch_size: int = 16
    img_size: int = 640
    workers: int = 8
    
    # 优化器参数
    lr0: float = 0.0005  # 降低初始学习率
    lrf: float = 0.001   # 最终学习率
    momentum: float = 0.937
    weight_decay: float = 0.0005
    
    # 训练设置
    device: str = "0"
    save_dir: str = "results/train_logs"
    project: str = "fish_detection"
    name: str = "yolov8l_fish_enhanced"
    
    # 增强参数 - 大幅增强，特别是对低性能类别
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    flipud: float = 0.1   # 增加上下翻转
    fliplr: float = 0.5
    mosaic: float = 0.8   # 增加mosaic概率
    mixup: float = 0.2    # 增加mixup概率
    copy_paste: float = 0.2  # 复制粘贴增强
    
    # 新增增强参数
    degrees: float = 20.0  # 增加旋转角度
    translate: float = 0.2  # 增加平移
    scale: float = 0.5
    shear: float = 5.0     # 增加剪切
    perspective: float = 0.001  # 透视变换
    
    # 损失函数权重
    box: float = 7.5
    cls: float = 0.8      # 增加分类损失权重
    dfl: float = 1.5
    
    # 早停和检查点
    patience: int = 100   # 大幅增加早停耐心值
    save_period: int = 10
    
    # 优化器选择
    optimizer: str = "AdamW"
    warmup_epochs: int = 5    # 增加热身epoch
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    
    # 类别权重 - 针对具体问题的调整
    class_weights: List[float] = field(default_factory=lambda: [
        3.0,  # Anabas_Testudineus (权重大幅增加)
        1.0,  # Batasio_Tengana
        1.2,  # Channa_Punctata
        3.0,  # Heteropneustes_Fossilis (权重大幅增加)
        1.5,  # Marcrobrachium_Malcoimsonii
        3.0,  # Mstacembelus_Armatus (权重大幅增加)
        1.2,  # Ompok_Bimaculatus
        0.5   # Puntius (降低权重，已表现良好)
    ])
    
    # 标签平滑
    label_smoothing: float = 0.15  # 增加标签平滑
    
    # Focal Loss参数
    ##fl_gamma: float = 1.5  # 使用focal loss处理难样本
    
    # 新的训练参数
    close_mosaic: int = 10  # 最后10个epoch关闭mosaic
    cos_lr: bool = True      # 使用余弦退火
    
    def __post_init__(self):
        # 确保class_weights长度正确
        if len(self.class_weights) != 8:
            self.class_weights = [1.0] * 8
    
    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if value is not None and not key.startswith('_'):
                result[key] = value
        return result