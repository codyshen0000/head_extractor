from PIL import Image
from head_extractor import ProcessorPipeline

print("--- 开始测试 head_extractor 包 ---")

# 1. 初始化 Pipeline
print("正在加载模型...")

pipeline = ProcessorPipeline.load()
print("模型加载成功！")


# 2. 读取一张待处理的图片 (请确保图片路径正确)
image_path = "./assets/001.jpg" # 示例图片路径
print(f"正在打开图片: {image_path}")
input_image = Image.open(image_path)


# 3. 提取头部
print("正在提取头部...")
# case:1
# default pad2square=True, output_mode='RGB'
extracted_head_image_rgb = pipeline.extract_head(input_image, output_mode='RGB')
extracted_head_image_rgb.save("./assets/001_head-default.webp")

# case:2
# pad2square=False
# rgba
result_rgba = pipeline.extract_head(
    input_image,
    output_mode='RGBA',
    pad2square=False
)
result_rgba.save("./assets/001_head-pad2square-false.webp")

# case:3
# black bg
result_black_bg = pipeline.extract_head(
    input_image,
    background_color=(0, 0, 0),
    pad2square=True,
    output_mode='RGB'
)
result_black_bg.save("./assets/001_head-black-bg.webp")



print(f"处理完成！结果已保存")
