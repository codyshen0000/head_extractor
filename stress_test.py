import time
import numpy as np
from PIL import Image
from head_extractor import ProcessorPipeline

def run_benchmark(
    image_path: str,
    num_warmup: int = 5,
    num_runs: int = 20,
    config: dict = None
):
    """
    对 head_extractor Pipeline 进行基准测试。

    Args:
        image_path (str): 用于测试的图片路径.
        num_warmup (int): 在正式测试前预热的次数.
        num_runs (int): 正式测试的运行次数.
        config (dict): 传递给 extract_head 的参数配置.
    """
    if config is None:
        config = {}

    print("--- Head Extractor 基准测试 ---")
    print(f"测试图片: {image_path}")
    print(f"预热次数: {num_warmup}")
    print(f"正式运行次数: {num_runs}")
    print(f"测试配置: {config}")
    print("-" * 30)

    # 1. 加载模型
    print("步骤 1: 正在加载模型...")
    load_start_time = time.time()
    try:
        pipeline = ProcessorPipeline.load()
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    load_end_time = time.time()
    print(f"模型加载完成，耗时: {load_end_time - load_start_time:.2f} 秒")

    # 2. 加载图片
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        print(f"错误：测试图片未找到，请检查路径 {image_path}")
        return

    # 3. 预热
    print(f"\n步骤 2: 正在进行 {num_warmup} 次预热...")
    for i in range(num_warmup):
        _ = pipeline.extract_head(image, **config)
        print(f"  预热 {i+1}/{num_warmup} 完成")
    print("预热完成。")

    # 4. 正式运行和计时
    print(f"\n步骤 3: 正在进行 {num_runs} 次正式测试...")
    timings = []
    for i in range(num_runs):
        start_time = time.time()
        _ = pipeline.extract_head(image, **config)
        end_time = time.time()
        duration = end_time - start_time
        timings.append(duration)
        print(f"  运行 {i+1}/{num_runs}，耗时: {duration:.4f} 秒")

    # 5. 计算并打印结果
    print("\n--- 基准测试结果 ---")
    total_time = sum(timings)
    avg_time = np.mean(timings)
    std_dev = np.std(timings)
    fps = 1.0 / avg_time if avg_time > 0 else 0

    print(f"总耗时 ({num_runs} 次运行): {total_time:.2f} 秒")
    print(f"平均每次耗时: {avg_time:.4f} 秒")
    print(f"标准差: {std_dev:.4f} 秒")
    print(f"处理速度 (FPS): {fps:.2f} 帧/秒")
    print("-" * 30)


if __name__ == '__main__':
    # --- 配置测试参数 ---
    # 请确保这张图片存在于项目的根目录下
    TEST_IMAGE_PATH = "./assets/001.jpg"

    # 场景1：测试默认配置 (RGB, 填充为正方形)
    default_config = {
        "crop_padding": 10,
        "background_color": (255, 255, 255),
        "pad2square": True,
        "output_mode": 'RGB'
    }
    run_benchmark(
        image_path=TEST_IMAGE_PATH,
        num_warmup=5,
        num_runs=20,
        config=default_config
    )

    # 测试其他场景
    # # 场景2：测试 RGBA 透明背景，不填充
    # print("\n\n") # 添加一些间隔
    # rgba_config = {
    #     "pad2square": False,
    #     "output_mode": 'RGBA'
    # }
    # run_benchmark(
    #     image_path=TEST_IMAGE_PATH,
    #     num_warmup=2,
    #     num_runs=10,
    #     config=rgba_config
    # )
