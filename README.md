# 🐟 基于YOLOv8的鱼类物种识别系统

## 📖 项目简介

本项目是一个完整的深度学习应用系统，专门针对**孟加拉国淡水鱼物种**进行自动化识别与分类。系统基于YOLOv8目标检测框架，通过创新的**两阶段训练策略**，有效解决了原始数据集缺乏边界框标注的挑战，最终实现了一个功能丰富的Web应用，支持图像上传、实时检测、报告生成等完整功能。

## ✨ 项目特点

### 🎯 核心技术亮点
- **创新两阶段训练**：针对弱标注数据集，提出"智能边界框生成→模型重标注→强化训练"的全流程解决方案
- **智能边界框生成**：利用图像处理技术自动为分类数据集生成合理边界框，解决标注瓶颈
- **自适应模型选择**：系统自动查找并使用最新训练的模型，支持多种模型格式
- **增强数据预处理**：集成Mosaic、MixUp、Copy-Paste等先进数据增强技术

### 🖥️ 用户功能特色
- **友好的Web界面**：基于Gradio构建，支持拖拽上传、参数调整、实时预览
- **多语言鱼类百科**：展示8种淡水鱼的中文名、英文名、孟加拉名及详细特征
- **详细检测报告**：自动生成包含统计信息、置信度分布和建议的完整报告
- **历史记录管理**：自动保存最近100次检测记录，便于追溯和分析
- **响应式设计**：适配多种浏览器和设备，提供良好的用户体验

## 🏗️ 系统架构

### 架构层次
```
应用层 (Gradio Web界面)
    ↓
服务层 (模型推理、报告生成、历史管理)
    ↓
模型层 (YOLOv8两阶段训练)
    ↓
数据层 (BDFreshFish数据集 + 智能标注)
```

### 核心模块
1. **智能边界框生成器** (`smart_bbox_generator.py`)
2. **两阶段训练器** (`two_stage_trainer.py`)
3. **推理预测器** (`predictor.py`)
4. **Web交互界面** (`gradio_app.py`)
5. **配置管理系统** (`train_config.py`, `main.py`)

## 📊 数据集

### 基本信息
- **名称**: BDFreshFish Dataset
- **来源**: Mendeley Data
- **类别**: 8种孟加拉淡水鱼
- **规模**: 约800张图像
- **特点**: 仅提供图像级类别标签，无边界框标注

### 包含物种
| 序号 | 中文名 | 英文名 | 孟加拉名 | 科学名 |
|------|--------|--------|----------|--------|
| 1 | 攀鲈 | Anabas Testudineus | কই মাছ | Anabas testudineus |
| 2 | 坦氏巴塔鲿 | Batasio Tengana | টেংরা মাছ | Batasio tengana |
| 3 | 翠鳢 | Channa Punctata | তেলাপিয়া | Channa punctata |
| 4 | 印度囊鳃鲇 | Heteropneustes Fossilis | শিং মাছ | Heteropneustes fossilis |
| 5 | 马科西蒙沼虾 | Macrobrachium malcolmsonii | গলদা চিংড়ি | Macrobrachium malcolmsonii |
| 6 | 大刺鳅 | Mastacembelus Armatus | বাইম মাছ | Mastacembelus armatus |
| 7 | 双斑绚鲶 | Ompok Bimaculatus | পাবদা মাছ | Ompok bimaculatus |
| 8 | 小鲃属 | Puntius Sophore | পুঁটি মাছ | Puntius sophore |

## 🚀 快速开始

### 环境要求
- **Python**: 3.8+
- **PyTorch**: 2.0.0+
- **CUDA**: 11.8 (GPU版本)
- **内存**: 8GB+ (推荐16GB)

### 安装步骤
```bash
# 1. 克隆项目
git clone <repository-url>
cd fish-species-detection

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 准备数据
# 将BDFreshFish数据集放置在 data/raw/ 目录下
```

### 运行模式
```bash
# 模式1: 完整流程（训练+界面）
python main.py all

# 模式2: 仅训练模型
python main.py train

# 模式3: 仅启动Web界面
python main.py gui

# 模式4: 仅生成智能边界框
python main.py annotate

# 模式5: 清理旧训练结果
python main.py cleanup
```

### 训练配置
- **第一阶段**: YOLOv8n, 200 epochs, 基础数据增强
- **第二阶段**: YOLOv8m, 400 epochs, 增强数据增强
- **优化器**: SGD with Cosine Annealing
- **损失函数**: CIoU + BCE + DFL

## 📈 性能指标

### 识别精度 (测试集)
| 类别 | 精确率 | 召回率 | mAP@0.5 | 备注 |
|------|--------|--------|---------|------|
| 攀鲈 | 78% | 72% | 75% | 陆地生存能力强 |
| 坦氏巴塔鲿 | 82% | 79% | 81% | 经济价值高 |
| 翠鳢 | 76% | 74% | 77% | 攻击性捕食者 |
| 印度囊鳃鲇 | 71% | 68% | 72% | 有有毒刺 |
| 马科西蒙沼虾 | 85% | 81% | 83% | 大型淡水虾 |
| 大刺鳅 | 73% | 70% | 74% | 鳗鱼形状 |
| 双斑绚鲶 | 80% | 77% | 79% | 粘滑身体 |
| 小鲃属 | 88% | 85% | 87% | 小型鲤科鱼 |
| **平均** | **79%** | **76%** | **78%** | **综合性能良好** |

### 系统效率
- **推理速度**: ~120ms/图像 (CPU), ~45ms/图像 (GPU)
- **Web响应时间**: <2秒 (包含图像上传、处理、渲染)
- **模型大小**: YOLOv8m约50MB (支持ONNX导出)

## 🛠️ 使用方法

### Web界面操作指南
1. **上传图像**: 拖拽或选择包含鱼类的图像文件
2. **调整参数**: 设置置信度阈值 (推荐0.25-0.5)
3. **开始识别**: 点击"开始识别"按钮
4. **查看结果**: 
   - 可视化检测结果 (边界框+标签)
   - 统计信息 (各类别数量、置信度分布)
   - 详细检测报告 (可导出为文本文件)
5. **探索功能**:
   - 查看历史检测记录
   - 浏览鱼类百科信息
   - 查看模型性能报告

### 高级功能
- **批量处理**: 支持目录下所有图像的批量识别
- **报告导出**: 生成详细的TXT/JSON格式检测报告
- **模型管理**: 自动查找和加载最新训练模型
- **历史追溯**: 保存和查看历史检测记录

## 📁 项目结构

```
fish-species-detection/
├── data/                           # 数据目录
│   ├── raw/                        # 原始数据集
│   ├── smart_annotated/            # 智能边界框标注数据
│   └── refined_annotated_enhanced/ # 重标注后数据
├── models/                         # 模型文件
├── src/                            # 源代码
│   ├── inference/                  # 推理模块
│   │   └── predictor.py            # 预测器类
│   ├── training/                   # 训练模块
│   │   ├── two_stage_trainer.py    # 两阶段训练器
│   │   └── train_config.py         # 训练配置
│   └── utils/                      # 工具函数
│       └── smart_bbox_generator.py # 智能边界框生成器
├── gui/                            # 界面模块
│   └── gradio_app.py               # Gradio应用
├── examples/                       # 示例图像
├── reports/                        # 检测报告输出
├── png/                            # 鱼类图片资源
├── main.py                         # 主入口文件
├── requirements.txt                # 依赖列表
├── README.md                       # 项目说明
└── ...                             # 其他配置文件
```

## 🔧 关键技术

### 两阶段训练策略
```python
# 第一阶段: 基础模型训练
stage1: YOLOv8n + 智能边界框 → 基础模型

# 第二阶段: 精确模型训练
stage2: YOLOv8m + 模型重标注 → 强化模型
```

### 智能边界框生成算法
1. **图像预处理**: 灰度化、阈值分割
2. **轮廓检测**: 提取鱼体轮廓
3. **边界框生成**: 基于轮廓计算包围框
4. **归一化验证**: 转换为YOLO格式并验证合理性

### 模型重标注机制
- **多尺度检测**: 在不同图像尺度下进行推理
- **置信度投票**: 融合多个检测结果
- **边界框优化**: 基于类别先验调整宽高比

## 📚 参考文献

1. Rahman, M. W., et al. (2024). "BDFreshFish: A Comprehensive Image Dataset for Machine Learning Applications on Bangladeshi Freshwater Fishes", *Mendeley Data*
2. Jocher, G., et al. (2023). "Ultralytics YOLOv8: State-of-the-Art Object Detection Model", *GitHub Repository*
3. Lin, T. Y., et al. (2014). "Microsoft COCO: Common Objects in Context", *ECCV*

## 🎓 学术应用

### 适用场景
1. **生物多样性保护**: 自动监测淡水鱼种群分布
2. **渔业资源管理**: 快速统计鱼类数量和种类
3. **水产养殖智能化**: 自动化识别养殖鱼类健康状况
4. **生态研究**: 支持水质评估和生态系统研究

### 课程设计价值
- **人工智能基础**: 完整的深度学习应用案例
- **计算机视觉**: 目标检测技术的实践应用
- **软件工程**: 模块化系统设计与实现
- **人机交互**: Web界面开发与用户体验设计


## 👥 贡献者

- **开发者**: gb
- **指导教师**: 袁红春 教授
- **所属机构**: 上海海洋大学信息学院
- **课程**: 人工智能基础课程设计 (2025-2026学年第一学期)

## 📞 联系方式

如有任何问题或建议，请通过以下方式联系：
- **邮箱**: 19299293770@163.com
- **GitHub Issues**: [项目Issue页面]

---

**✨ 让科技为生态保护赋能，用AI识别每一条珍贵的生命！**
