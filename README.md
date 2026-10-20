智慧化移动工具生产要素工程车辆及人员检测数据集
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bf92adccdc904c6dbda7d982e66a0250.png)

**

**智慧工地数据集，工程车辆及人员检测 包含13种工地中主要移动目标，1500张左右 txt格式， 已划分train：val：test 。yolo可以直接训练 'Bulldozer' - 推土机 'Concrete mixer' - 混凝土搅拌机 'Crane' - 起重机 'Excavator' - 挖掘机 'Hanging head' - 吊装头 'Loader' - 装载机 'Other vehicle' - 其他车辆 'Pile driving' - 打桩机 'Pump truck' - 泵车 'Roller' - 压路机 'Static crane' - 塔吊 'Truck' - 卡车 'Worker' - 工人，**

**
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/15401ebbd8344743a8b009ef34185762.png)


---

# 🏗️ 智慧工地工程车辆及人员检测数据集（YOLO 格式）

> ✅ 支持 YOLOv11 直接训练  
> ✅ 包含 13 类工地常见目标  
> ✅ 约 1500 张高清图像，`txt` 标注格式  
> ✅ 已划分：`train` / `val` / `test`  
> ✅ 适用于智能监控、安全预警、AI 巡检等场景

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/83270c9739d44090b2850b3f7294ef5a.png)

## 📊 数据集统计表

| 类别英文名 | 类别中文名 | 图像中出现数量（约） | 总标注框数（约） | 备注 |
|------------|-------------|------------------|----------------|------|
| `Bulldozer` | 推土机 | 420 | 480 | 常见于土方作业区 |
| `Concrete mixer` | 混凝土搅拌机 | 560 | 630 | 流动式搅拌车或固定设备 |
| `Crane` | 起重机（汽车吊） | 390 | 410 | 移动式起重设备 |
| `Excavator` | 挖掘机 | 680 | 750 | 最常见的挖掘机械 |
| `Hanging head` | 吊装头 | 310 | 330 | 高空吊装关键部件 |
| `Loader` | 装载机 | 400 | 440 | 装卸物料用 |
| `Other vehicle` | 其他车辆 | 350 | 370 | 不属于上述类别的工程车 |
| `Pile driving` | 打桩机 | 280 | 300 | 地基施工专用机械 |
| `Pump truck` | 泵车 | 300 | 320 | 混凝土输送泵车 |
| `Roller` | 压路机 | 330 | 360 | 路面压实作业 |
| `Static crane` | 塔吊 | 450 | 480 | 固定式高层起重设备 |
| `Truck` | 卡车 | 620 | 680 | 运输建材、渣土等 |
| `Worker` | 工人 | 1200 | 2100+ | 出现频率最高，分布广泛 |

> 🔍 **总计**：
- **图像总数**：约 **1500 张**
- **训练集（train）**：~1200 张
- **验证集（val）**：~150 张
- **测试集（test）**：~150 张
- **总标注框数**：约 **6,000+**

> ⚠️ 注：部分图像含多个目标，尤其“工人”常密集出现。

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/08c8f9b7a1df4459b1def21e3df60c01.png)

## 🗂️ 推荐目录结构

```bash
construction_site_dataset/
├── images/
│   ├── train/          # ~1200 张
│   ├── val/            # ~150 张
│   └── test/           # ~150 张
├── labels/
│   ├── train/          # 对应 .txt 文件（YOLO 格式）
│   ├── val/
│   └── test/
├── construction.yaml   # YOLO 配置文件
└── train.py            # YOLOv11 训练脚本
```

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2495e36987e44e69bcf3cfc4d55e7380.png)

## 🛠️ YOLO 配置文件：`construction.yaml`

```yaml
# construction.yaml
train: ./construction_site_dataset/images/train
val: ./construction_site_dataset/images/val
test: ./construction_site_dataset/images/test

nc: 13
names: 
  - Bulldozer
  - Concrete mixer
  - Crane
  - Excavator
  - Hanging head
  - Loader
  - Other vehicle
  - Pile driving
  - Pump truck
  - Roller
  - Static crane
  - Truck
  - Worker
```

> ✅ 将此文件保存在项目根目录下。

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bc71dd422bdd4915a0f3c207dbfb4fc9.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/be6a2a1fdafa424f9658747d8a80b4f7.png)


## 🚀 YOLOv11 完整训练代码：`train.py`

```python
# train.py
from ultralytics import YOLO
import os
import time

# ======================= 参数设置 =======================
MODEL_NAME = "yolov11s.pt"               # 推荐使用 yolov11n/s/m；小数据可用 s 或 m
CONFIG_FILE = "construction.yaml"        # 数据集配置路径
EPOCHS = 150
BATCH_SIZE = 16                          # 若显存不足可设为 8 或 4
IMG_SIZE = 640                           # 输入尺寸（也可尝试 1280 提升小目标检测能力）
PROJECT_NAME = "smart_construction_detection"
EXPERIMENT_NAME = "exp_construction_yolo11"

DEVICE = 0 if os.environ.get("CUDA_AVAILABLE") else "cpu"  # 自动选择 GPU/CPU

# 检查配置文件是否存在
if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(f"[ERROR] 找不到配置文件: {CONFIG_FILE}")

# ======================= 加载模型 =======================
print("🚀 加载 YOLOv11 模型...")
model = YOLO(MODEL_NAME)  # 自动下载预训练权重

# ======================= 开始训练 =======================
print("🔥 开始训练智慧工地目标检测模型...")
start_time = time.time()

results = model.train(
    data=CONFIG_FILE,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    imgsz=IMG_SIZE,
    device=DEVICE,
    project=PROJECT_NAME,
    name=EXPERIMENT_NAME,
    exist_ok=True,
    patience=40,                         # 早停机制：验证损失连续40轮不降则停止
    save=True,
    save_period=10,                      # 每10轮保存一次检查点
    cache=False,                         # 设置为 True 可加快加载速度（需足够内存）
    workers=4,

    # 优化器设置
    optimizer='AdamW',
    lr0=0.001,                           # 初始学习率
    lrf=0.01,                            # 最终学习率 = lr0 * lrf
    momentum=0.937,
    weight_decay=0.0005,

    # 数据增强策略（适合复杂工地场景）
    hsv_h=0.015,                         # 色调扰动
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=15.0,                        # 允许轻微旋转（模拟摄像头角度变化）
    translate=0.2,
    scale=0.6,
    shear=2.0,
    flipud=0.0,                          # 上下翻转不合理（天空/地面不对称）
    fliplr=0.5,                          # 左右翻转增强
    mosaic=1.0,                          # Mosaic 增强提升遮挡和小目标鲁棒性
    mixup=0.1,                           # MixUp 缓解过拟合
    copy_paste=0.15,                     # 对稀有类如 Pile driving、Hanging head 有效
)

# ======================= 输出训练耗时 =======================
training_time = (time.time() - start_time) / 60
print(f"✅ 训练完成！总耗时: {training_time:.2f} 分钟")
print(f"📍 模型保存路径: {results.save_dir}")

# ======================= 验证模型性能 =======================
print("🔍 开始验证...")
metrics = model.val()
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.p:.4f}")
print(f"Recall: {metrics.box.r:.4f}")

# 显示各类别 AP @0.5
print("\n📊 各类别 mAP@0.5:")
for i, cls_name in enumerate(model.names):
    print(f"  {cls_name}: {metrics.box.maps[i]:.4f}")

# ======================= 导出 ONNX 模型（便于部署） =======================
print("📦 导出 ONNX 模型...")
onnx_path = model.export(format="onnx", dynamic=True, simplify=True)
print(f"ONNX 模型已保存至: {onnx_path}")
```

---

## 🧪 推理示例代码：`inference.py`

```python
# inference.py
from ultralytics import YOLO
import cv2

# 加载训练好的模型
model = YOLO("smart_construction_detection/exp_construction_yolo11/weights/best.pt")

# 单张图片推理
image_path = "demo.jpg"
results = model(image_path, conf=0.4, iou=0.5, imgsz=640)

# 可视化结果
for r in results:
    im_array = r.plot()  # 绘制边界框和标签
    im = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
    cv2.imshow("Construction Site Detection", im)
    cv2.waitKey(0)
    cv2.imwrite("result_detected.jpg", im)

# 打印检测详情
for r in results:
    boxes = r.boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]
        print(f"检测到: {class_name} (置信度: {conf:.2f})")
```

---

## 🎯 应用场景建议

| 场景 | 功能实现 |
|------|----------|
| 🚧 **安全区域入侵检测** | 检测非工作人员 (`Worker`) 进入危险区 |
| 🏗️ **机械设备运行监控** | 实时识别 `Crane`, `Excavator` 等是否违规操作 |
| 🧱 **吊装作业安全监管** | 联合检测 `Hanging head` + `Worker` 距离，防止高空坠物伤人 |
| 🪖 **安全帽佩戴检测** | 在 `Worker` 检测基础上叠加头盔分类（可扩展） |
| 📊 **施工进度分析** | 统计不同类型设备出现频次，评估工程进度 |
| 🤖 **无人巡检机器人** | 部署于巡逻机器人，自动上报异常行为 |

---



---


