标注指南（仅按严重度）

目标：为每个可见的损伤区域画一个边界框，并仅标注严重度（0: minor, 1: moderate, 2: severe）。最终导出为 YOLO 格式（每张图片对应 `.txt` 文件）。

类映射：
- 0: minor
- 1: moderate
- 2: severe

YOLO 标签格式（每行）：
<class> <x_center> <y_center> <width> <height>
所有坐标均为相对于图片宽高的归一化浮点数，范围 [0,1]。

示例（图片宽 1024，高 768，bbox 像素坐标 x_min=100,y_min=150,x_max=300,y_max=450，且 class=1）：
- x_center = (100+300)/2 / 1024 = 0.1953
- y_center = (150+450)/2 / 768 = 0.3906
- width = (300-100)/1024 = 0.1953
- height = (450-150)/768 = 0.3906
标签行：
1 0.1953 0.3906 0.1953 0.3906

目录结构建议（YOLOv8 训练常见）：
- data/images/train
- data/images/val
- data/images/test
- data/labels/train
- data/labels/val
- data/labels/test

标注流程建议（半自动 + 人工校正）：
1) 用工具（LabelImg/CVAT）手动标注并导出为 PascalVOC 或 CSV，或在表格里记录像素 bbox。
2) 使用仓库内的 `tools/convert_bbox_to_yolo.py` 将 CSV/像素坐标转换为 YOLO `.txt`。
3) 把生成的 labels 放到 `data/labels/...` 与图片对应的路径下。
4) 创建 `data.yaml` 指向图片路径与 `names`（类名），然后用 Ultralytics/YOLOv8 训练。

CSV 示例格式（必须包含 header，列名示例）：
filename,x_min,y_min,x_max,y_max,class
000001.jpg,100,150,300,450,1

注意事项：
- 若某张图片没有 bbox，仍应创建空的 `.txt` 文件（或确保训练时被正确识别为无标签）。
- 保持标注一致性：尽量框住整个损伤区域，避免把多个相连的碎片分开标注（合并为一个框）。
- 可以设置最小可接受尺寸（例如短边 >= 16 px）过滤噪声。

快速命令示例：
生成 labels：
python tools/convert_bbox_to_yolo.py --csv annotations/bboxes.csv --images-dir data/images --labels-dir data/labels --create-empty
