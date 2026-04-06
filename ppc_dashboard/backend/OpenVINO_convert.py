import os
import shutil
import yaml
from ultralytics import YOLO

# ==============================================================
# 🔥 核心修复：猴子补丁 (Monkey Patch) 拦截 NNCF 严格节点验证
# 解决 "Ignored nodes... were not found in the NNCFGraph" 报错
# ==============================================================
import nncf
original_ignored_scope_init = nncf.IgnoredScope.__init__

def patched_ignored_scope_init(self, *args, **kwargs):
    kwargs['validate'] = False  # 强制关闭节点验证，找不到特定层也绝不报错！
    original_ignored_scope_init(self, *args, **kwargs)

nncf.IgnoredScope.__init__ = patched_ignored_scope_init
# ==============================================================

def optimize_yolo_for_intel_cpu_int8_calib():
    model_path = "model/best_3_3.pt"  # 您的 pt 模型路径

    # 🔥 您找到的校准集绝对路径
    calib_data_dir = r"D:\Desktop\PET1\model\testset"

    print("🛠️ 正在读取模型类别并生成临时校准配置文件...")
    temp_model = YOLO(model_path)
    class_names = temp_model.names

    calib_yaml_path = "temp_calib.yaml"
    calib_yaml_content = {
        "path": calib_data_dir,
        "train": "val/images",
        "val": "val/images",
        "names": class_names
    }

    with open(calib_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(calib_yaml_content, f, allow_unicode=True)

    # 自动推断 Ultralytics 导出的默认文件夹名称 (适配 best_3_3)
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    export_dir_name_int8 = f"model/{base_name}_int8_openvino_model"
    export_dir_name_fp32 = f"model/{base_name}_openvino_model"

    # ==========================================
    # 1. 导出 480x480 FP32 原生模型 (老 CPU 保命用)
    # ==========================================
    print("\n🔄 [1/4] 正在导出 480x480 FP32 原生模型...")
    model_480_fp32 = YOLO(model_path)
    model_480_fp32.export(format="openvino", imgsz=480, dynamic=False)

    target_dir_480_fp32 = f"model/{base_name}_openvino_model_480_fp32"
    if os.path.exists(target_dir_480_fp32):
        shutil.rmtree(target_dir_480_fp32)
    if os.path.exists(export_dir_name_fp32):
        os.rename(export_dir_name_fp32, target_dir_480_fp32)
    print(f"✅ 480p FP32 原生模型已保存至: {target_dir_480_fp32}")

    # ==========================================
    # 2. 导出 640x640 FP32 原生模型 (老 CPU 保命用)
    # ==========================================
    print("\n🔄 [2/4] 正在导出 640x640 FP32 原生模型...")
    model_640_fp32 = YOLO(model_path)
    model_640_fp32.export(format="openvino", imgsz=640, dynamic=False)

    target_dir_640_fp32 = f"model/{base_name}_openvino_model_640_fp32"
    if os.path.exists(target_dir_640_fp32):
        shutil.rmtree(target_dir_640_fp32)
    if os.path.exists(export_dir_name_fp32):
        os.rename(export_dir_name_fp32, target_dir_640_fp32)
    print(f"✅ 640p FP32 原生模型已保存至: {target_dir_640_fp32}")

    # ==========================================
    # 3. 导出 480x480 静态 INT8 模型 (带校准，新 CPU 加速用)
    # ==========================================
    print("\n🔄 [3/4] 正在使用真实测试集进行 480x480 INT8 量化校准 (此过程需要几分钟以运算激活值范围)...")
    model_480_int8 = YOLO(model_path)
    model_480_int8.export(format="openvino", imgsz=480, int8=True, data=calib_yaml_path, simplify=False, dynamic=False)

    target_dir_480_int8 = f"model/{base_name}_openvino_model_480_int8"
    if os.path.exists(target_dir_480_int8):
        shutil.rmtree(target_dir_480_int8)

    if os.path.exists(export_dir_name_int8):
        os.rename(export_dir_name_int8, target_dir_480_int8)
    elif os.path.exists(export_dir_name_fp32):
        os.rename(export_dir_name_fp32, target_dir_480_int8)
    print(f"✅ 480p INT8 静态校准模型已保存至: {target_dir_480_int8}")

    # ==========================================
    # 4. 导出 640x640 静态 INT8 模型 (带校准，新 CPU 加速用)
    # ==========================================
    print("\n🔄 [4/4] 正在使用真实测试集进行 640x640 INT8 量化校准...")
    model_640_int8 = YOLO(model_path)
    model_640_int8.export(format="openvino", imgsz=640, int8=True, data=calib_yaml_path, simplify=False, dynamic=False)

    target_dir_640_int8 = f"model/{base_name}_openvino_model_640_int8"
    if os.path.exists(target_dir_640_int8):
        shutil.rmtree(target_dir_640_int8)

    if os.path.exists(export_dir_name_int8):
        os.rename(export_dir_name_int8, target_dir_640_int8)
    elif os.path.exists(export_dir_name_fp32):
        os.rename(export_dir_name_fp32, target_dir_640_int8)
    print(f"✅ 640p INT8 静态校准模型已保存至: {target_dir_640_int8}")

    # ==========================================
    # 清理作业
    # ==========================================
    if os.path.exists(calib_yaml_path):
        os.remove(calib_yaml_path)

    print("\n🎉 恭喜！FP32 与 INT8 双格式、双解析度 (共 4 个模型) 导出完成！")
    print("您现在可以挂载到后端引擎，系统将根据硬件层级自动抽取最合适的模型享受满血性能。")

if __name__ == "__main__":
    optimize_yolo_for_intel_cpu_int8_calib()