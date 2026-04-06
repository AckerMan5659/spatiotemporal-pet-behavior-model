import os
import sys
import argparse

# 1) 自动将项目根目录加入 Python 路径，确保能导入 scripts.pipeline
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.pipeline_gru import run   # 2) 直接导入我们之前写好的主流程函数

def main():
    parser = argparse.ArgumentParser(description="Run YOLO + EfficientFormer/EfficientViT/RepViT Behavior Pipeline")
    parser.add_argument("--cfg", default="config.yaml", help="Path to config file")
    parser.add_argument("--video", required=True, help="Path to the input video file")
    # 修改这里的描述，说明传入的是文件夹
    parser.add_argument("--save", default="output/", help="Directory to save output video, e.g. output/")
    # parser.add_argument("--show_bbox", action="store_true", help="Whether to show bounding boxes in the output video")
    
    args = parser.parse_args()

    print("✅ Starting Behavior Recognition Pipeline...")
    print(f"➡ Config file: {args.cfg}")
    print(f"➡ Input video: {args.video}")
    if args.save:
        print(f"➡ Output directory will be: {args.save}")
    else:
        print("➡ No save path provided, will only display results.")
    
    # print(f"➡ Show Bounding Box: {'Yes' if args.show_bbox else 'No'}")


    # 3) 调用 pipeline 主流程时，使用关键字参数绑定变量，避免错位
    run(cfg_path=args.cfg, video_path=args.video, save_path=args.save)

if __name__ == "__main__":
    main()