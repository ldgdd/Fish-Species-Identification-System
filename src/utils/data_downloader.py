import os
import zipfile
import requests
from tqdm import tqdm

class DatasetDownloader:
    def __init__(self, data_dir="data/raw"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def download_bdfreshfish(self, url=None):
        """
        下载BDFreshFish数据集
        数据集URL: https://data.mendeley.com/datasets/29kjy99kkh/1
        """
        # 这里需要手动下载，因为需要同意Mendeley的条款
        print("请手动下载数据集:")
        print("1. 访问: https://data.mendeley.com/datasets/29kjy99kkh/1")
        print("2. 下载 'BDFreshFish dataset.zip'")
        print("3. 将文件放置在 data/raw/ 目录下")
        
    def extract_dataset(self, zip_path):
        """解压数据集"""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        print(f"数据集已解压到 {self.data_dir}")
        
    def organize_dataset_structure(self):
        """组织数据集结构"""
        # 创建YOLO格式的目录结构
        yolo_structure = {
            "images": ["train", "val", "test"],
            "labels": ["train", "val", "test"]
        }
        
        for main_dir, sub_dirs in yolo_structure.items():
            for sub_dir in sub_dirs:
                path = os.path.join("data/processed", main_dir, sub_dir)
                os.makedirs(path, exist_ok=True)
                
        print("YOLO目录结构创建完成")

if __name__ == "__main__":
    downloader = DatasetDownloader()
    downloader.organize_dataset_structure()