# Head Extractor

## 🚀 安装

bash setup.sh 

如果出现：
--- 步骤 6: 验证安装 ---
✅ head_extractor
✅ mmengine
✅ mmdet (core)
✅ mmseg
✅ mmcv
✅ MMCV 版本: 2.1.0
则表示安装完成

测试：python test.py

## 💡 使用方法

以下是一个基本的使用示例，展示了如何加载模型并处理单张图片。

```python
# example.py
from PIL import Image
from head_extractor import ProcessorPipeline

# 1. 初始化 Pipeline，它会自动加载封装在包内的模型
# 这个过程在首次调用时可能会花费一些时间
print("正在加载模型...")
pipeline = ProcessorPipeline.load()
print("模型加载成功！")

# 2. 打开一张待处理的图片
try:
    input_image = Image.open("001.jpg") # 替换为你的图片路径
except FileNotFoundError:
    print("错误：测试图片未找到！")
    exit()

# 3. 提取头部
# 默认输出为带白色背景、填充为正方形的 RGB 图像
print("正在提取头部...")
extracted_head = pipeline.extract_head(input_image)

# 4. 保存结果
output_path = "extracted_head_default.png"
extracted_head.save(output_path)
print(f"处理完成！结果已保存至: {output_path}")
```

## ⚡️ 性能基准 (Performance Benchmark)

以下是在**单张**图片上进行处理的性能测试结果。

**测试环境**:

- **GPU**: NVIDIA H100
- **配置**: 默认参数 (`long_edge=1024, crop_padding=10, pad2square=True`)


| 指标               | 结果          |
| :------------------- | :-------------- |
| **平均每次耗时**   | `~0.27` 秒    |
| **处理速度 (FPS)** | `~3.75` 帧/秒 |
| **显存占用**       | `~7.7` GB     |

## 📚 API 参考

### `ProcessorPipeline.extract_head()`

```python
pipeline.extract_head(
    image: Image.Image,
    crop_padding: int = 10,
    background_color: tuple = (255, 255, 255),
    pad2square: bool = True,
    output_mode: str = 'RGB'
) -> Image.Image:
```

**参数**:

- `image` (`PIL.Image.Image`): **必需**。输入的 PIL 图像对象。
- `crop_padding` (`int`, 可选, 默认: `10`): 在检测到的头部边界框周围额外增加的边距（像素）。
- `background_color` (`tuple`, 可选, 默认: `(255, 255, 255)`): 当 `output_mode` 为 `'RGB'` 时，用于背景和填充区域的 RGB 颜色。
- `pad2square` (`bool`, 可选, 默认: `True`): 是否将最终的输出图像填充为正方形。
  - `True`: 输出为正方形图像。
  - `False`: 输出为紧密裁剪后的矩形图像。
- `output_mode` (`str`, 可选, 默认: `'RGB'`): 输出图像的模式。
  - `'RGB'`: 输出为三通道的 RGB 图像，背景为 `background_color` 指定的颜色。
  - `'RGBA'`: 输出为带 Alpha 通道的四通道 RGBA 图像，背景为透明。
- `long_edge` (`int`, 可选, 默认: `1024`): 在送入模型前，将图片长边缩放到的尺寸。


**返回值**:

- `PIL.Image.Image`: 处理完成后的 PIL 图像对象。

## ⚙️ 更多示例

#### 示例 1: 输出带透明背景、不填充为正方形的头像

```python
result_rgba = pipeline.extract_head(
    input_image,
    output_mode='RGBA',
    pad2square=False
)
# 注意：带透明通道的图像应保存为 PNG 格式
result_rgba.save("extracted_head_transparent.png")
```

#### 示例 2: 输出带黑色背景、填充为正方形的头像

```python
result_black_bg = pipeline.extract_head(
    input_image,
    background_color=(0, 0, 0),
    pad2square=True,
    output_mode='RGB'
)
result_black_bg.save("extracted_head_black_bg.jpg")
```

```

```
