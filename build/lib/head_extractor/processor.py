import os
import numpy as np
from mmseg.apis import inference_model, init_model
from PIL import Image
import cv2
from enum import Enum
import importlib.resources


'''
Labels: 
0: 'background'	1: 'top'	2: 'outer'	3: 'skirt'
4: 'dress'	5: 'pants'	6: 'leggings'	7: 'headwear'
8: 'eyeglass'	9: 'neckwear'	10: 'belt'	11: 'footwear'
12: 'bag'	13: 'hair'	14: 'face'	15: 'skin'
16: 'ring'	17: 'wrist_wearing'	18: 'socks'	19: 'gloves'
20: 'necklace'	21: 'rompers'	22: 'earrings'	23: 'tie'
24: Left_Foot
25: Left_Hand
26: Left_Lower_Arm
27: Left_Lower_Leg
28: Left_Upper_Arm
29: Left_Upper_Leg
30: Right_Foot
31: Right_Hand
32: Right_Lower_Arm
33: Right_Lower_Leg
34: Right_Upper_Arm
35: Right_Upper_Leg
36: Torso
'''

class PersonSeg:
    def __init__(self, config_path, model_path, device='cuda'):
        # init model
        self.model = init_model(config_path, model_path, device=device)

    def process(self, image):
        result = inference_model(self.model, image)
        pred_seg = result.pred_sem_seg.data.cpu().numpy()[0]
        return pred_seg

class TaskType(Enum):
    face = "face"
    head = "head"
    head_plus_shoulders = "head_plus_shoulders"

    # 衣服相关任务
    top_cloth = "top_cloth"
    bottom_cloth = "bottom_cloth"
    full_clothes = "full_clothes"

    # 全身相关任务
    full_character = "full_character"

class ProcessorPipeline:
    """
    该功能主要用于从单个图像中提取指定内容的mask
    """
    def __init__(self, seg_pipe: PersonSeg):
        self.seg_pipe = seg_pipe

    @classmethod
    def load(cls, device: str = 'cuda') -> "ProcessorPipeline":
        """
        从包内加载模型和配置来初始化 Pipeline。
        不再需要外部路径。
        """
        # 使用 importlib.resources 安全地获取包内文件的路径
        with importlib.resources.path('head_extractor.models', 'depth_anything_large_mask2former_16xb1_160k_human_parsing_fashion_1024x1024.py') as config_path:
            with importlib.resources.path('head_extractor.models', 'ckpt.pth') as model_path:
                seg_pipe = PersonSeg(str(config_path), str(model_path), device=device)
        
        return cls(seg_pipe)

    def process(
        self,
        image: Image.Image,
        task_type: TaskType,
        long_edge: int = 1024
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        从图像中提取mask，内部流程优化为返回NumPy数组。
        
        Args:
            image: 输入图像
            task_type: 任务类型 ('head' or 'face')
            long_edge (int): 用于缩放图像的长边尺寸，值越小速度越快。
        
        Returns:
            (处理后的图像 NumPy 数组, 生成的mask NumPy 数组)
        """
        # 1. 预处理图像：统一转换为numpy array (RGB)
        if isinstance(image, Image.Image):
            image_np = np.array(image.convert("RGB"))
        else: # 假设是numpy array
            image_np = image
        
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

        processed_image_np = self.resize_long_edge(image_np, long_edge=long_edge)
        ori_h, ori_w = processed_image_np.shape[:2]

        # 2. 运行分割
        pred_mask_map = self.seg_pipe.process(processed_image_np)

        if task_type == TaskType.head_plus_shoulders:
            # 2.1 先做“头部”基础mask
            head_labels = [7, 8, 13, 14]  # headwear, eyeglass, hair, face
            head_mask = np.isin(pred_mask_map, head_labels).astype(np.float32)
            head_mask = cv2.resize(head_mask, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)

            # 2.2 计算头部bbox并向下和左右扩展一段
            rows = np.any(head_mask > 0, axis=1)
            cols = np.any(head_mask > 0, axis=0)
            if np.any(rows) and np.any(cols):
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                h_box = max(1, rmax - rmin)
                w_box = max(1, cmax - cmin)

                down_ratio = 0.1   # 向下扩展比例（相对头bbox高）
                side_ratio = 0.6  # 左右扩展比例（相对头bbox宽）

                r2max = min(ori_h, rmax + int(h_box * down_ratio))
                c2min = max(0, cmin - int(w_box * side_ratio))
                c2max = min(ori_w, cmax + int(w_box * side_ratio))

                rect_mask = np.zeros((ori_h, ori_w), dtype=np.float32)
                rect_mask[rmin:r2max, c2min:c2max] = 1.0

                # 2.3 在扩展矩形内，仅保留“人物相关像素”（过滤掉背景）
                person_labels = list(range(1, 37))  # 1..36 都是人物部件
                person_mask = np.isin(pred_mask_map, person_labels).astype(np.float32)
                person_mask = cv2.resize(person_mask, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)

                initial_mask = np.clip(head_mask + (person_mask * rect_mask), 0, 1)
            else:
                initial_mask = head_mask
        else:
            # 其它任务保持原逻辑
            labels_map = self._get_labels_for_task(task_type)
            primary_labels = labels_map['primary']
            initial_mask = np.isin(pred_mask_map, primary_labels).astype(np.float32)
            initial_mask = cv2.resize(initial_mask, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)

        # 3. 后处理（不同任务的形态学策略）
        final_mask_np = self._apply_task_specific_mask_processing(initial_mask, task_type, ori_h, ori_w)

        # 4. 返回
        final_mask_uint8 = (final_mask_np * 255).astype(np.uint8)
        return processed_image_np, final_mask_uint8

    def _get_labels_for_task(self, task_type: TaskType) -> dict:
        """根据任务类型获取对应的标签映射"""
        labels_map = {
            TaskType.face: { 'primary': [8, 14] }, # eyeglass, face
            TaskType.head: { 'primary': [7, 8, 13, 14] }, # headwear, eyeglass, hair, face
            TaskType.top_cloth: { 'primary': [1, 2] }, # top, outer
            TaskType.bottom_cloth: { 'primary': [3, 4, 5, 6] }, # skirt, dress, pants, leggings
            TaskType.full_clothes: { 'primary': [1, 2, 3, 4, 5, 6] }, # all clothes
            TaskType.full_character: { 'primary': list(range(1, 37)) }, # 包含所有人物相关部分
        }
        return labels_map.get(task_type, {'primary': []})

    def _apply_task_specific_mask_processing(self, mask: np.ndarray, task_type: TaskType, ori_h: int, ori_w: int) -> np.ndarray:
        """根据任务类型对mask进行特殊处理"""
        if task_type == TaskType.face:
            # 人脸任务：简单膨胀
            expand_kernel = 5
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (expand_kernel, expand_kernel))
            mask = cv2.dilate((mask > 0.5).astype(np.float32), kernel)
            
        elif task_type == TaskType.head:
            # 头部任务：先腐蚀再膨胀
            kernel = np.ones((7, 7), dtype=np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            
            expand_kernel = 11
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (expand_kernel, expand_kernel))
            mask = cv2.dilate((mask > 0.5).astype(np.float32), kernel)

        elif task_type == TaskType.head_plus_shoulders:
            # 比 head 更偏向“向下与左右扩展”的膨胀（高度核 > 宽度核）
            # 轻微腐蚀，避免边界毛刺
            erode_k = 5
            kernel = np.ones((erode_k, erode_k), dtype=np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)

            max_side = max(ori_h, ori_w)
            h_kernel = max(15, int(max_side * 0.05))  # 更高
            w_kernel = max(11, int(max_side * 0.03))  # 稍窄
            # 保证奇数
            h_kernel = h_kernel // 2 * 2 + 1
            w_kernel = w_kernel // 2 * 2 + 1

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w_kernel, h_kernel))
            mask = cv2.dilate((mask > 0.5).astype(np.float32), kernel)

        if task_type in [TaskType.top_cloth, TaskType.bottom_cloth, TaskType.full_clothes, TaskType.full_character]:
            # 衣服相关任务：膨胀和模糊处理
            expand_ratio = 0.01
            max_side = max(ori_h, ori_w)
            blur_kernel = 1
            expand_kernel = int(max_side * expand_ratio) // 2 * 2 + 1
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (expand_kernel, expand_kernel))
            expanded = cv2.dilate((mask > 0.5).astype(np.uint8), kernel)
            
            blurred = cv2.GaussianBlur(
                expanded.astype(np.float32),
                (blur_kernel, blur_kernel),
                sigmaX=0,
            )
            mask = np.clip(blurred / (blurred.max() + 1e-6), 0, 1)
            
        return mask

    @staticmethod
    def resize_long_edge(image_np: np.ndarray, long_edge=1024) -> np.ndarray:
        """将图像等比例缩放到指定长边尺寸 (使用OpenCV)"""
        original_height, original_width = image_np.shape[:2]
        
        max_dimension = max(original_width, original_height)
        if max_dimension <= long_edge:
            return image_np
        
        ratio = long_edge / max_dimension
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        
        # 使用cv2.INTER_AREA进行缩放，对于缩小图像效果较好且速度快
        return cv2.resize(image_np, (new_width, new_height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _pad_to_square_np(image_np: np.ndarray, background_value: tuple) -> np.ndarray:
        """将NumPy图像填充为正方形"""
        height, width = image_np.shape[:2]
        if width == height:
            return image_np
        
        max_dim = max(width, height)
        
        # 根据通道数确定背景色
        channels = image_np.shape[2] if len(image_np.shape) > 2 else 1
        
        # 创建一个正确尺寸的背景板
        padded_image = np.full((max_dim, max_dim, channels), background_value, dtype=image_np.dtype)

        paste_x = (max_dim - width) // 2
        paste_y = (max_dim - height) // 2
        
        padded_image[paste_y:paste_y+height, paste_x:paste_x+width] = image_np
        return padded_image

    @staticmethod
    def pad_to_square(image: Image.Image, background_color: tuple = (255, 255, 255)) -> Image.Image:
        """
        将图像填充为正方形
        
        Args:
            image: 输入图像
            background_color: 填充的背景颜色
        
        Returns:
            填充为正方形的图像
        """
        width, height = image.size
        if width == height:
            return image
        
        max_dim = max(width, height)
        padded_image = Image.new(image.mode, (max_dim, max_dim), background_color)
        paste_x = (max_dim - width) // 2
        paste_y = (max_dim - height) // 2
        padded_image.paste(image, (paste_x, paste_y))
        return padded_image

    def crop_image_by_mask(self, image: Image.Image, mask: Image.Image, padding: int = 20) -> Image.Image:
        """
        根据mask裁剪图像，只保留mask覆盖的区域
        
        Args:
            image: 原始图像
            mask: 二值mask图像
            padding: 裁剪区域的边距扩展像素数
        
        Returns:
            裁剪后的图像
        """
        # 转换为numpy数组
        mask_np = np.array(mask)
        image_np = np.array(image)
        
        # 找到mask中非零像素的边界框
        rows = np.any(mask_np > 0, axis=1)
        cols = np.any(mask_np > 0, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            # 如果mask为空，返回原图
            return image
        
        # 获取边界框坐标
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # 添加padding并确保不超出图像边界
        h, w = image_np.shape[:2]
        rmin = max(0, rmin - padding)
        rmax = min(h, rmax + padding + 1)
        cmin = max(0, cmin - padding)
        cmax = min(w, cmax + padding + 1)
        
        # 裁剪图像
        cropped_image = image_np[rmin:rmax, cmin:cmax]
        
        return Image.fromarray(cropped_image)

    def _crop_image_and_mask_np(self, image_np: np.ndarray, mask_np: np.ndarray, padding: int = 20) -> tuple[np.ndarray, np.ndarray]:
        """根据mask同时裁剪NumPy图像和mask"""
        rows = np.any(mask_np > 0, axis=1)
        cols = np.any(mask_np > 0, axis=0)

        if not np.any(rows) or not np.any(cols):
            return image_np, mask_np

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        h, w = image_np.shape[:2]
        rmin = max(0, rmin - padding)
        rmax = min(h, rmax + padding + 1)
        cmin = max(0, cmin - padding)
        cmax = min(w, cmax + padding + 1)

        cropped_image_np = image_np[rmin:rmax, cmin:cmax]
        cropped_mask_np = mask_np[rmin:rmax, cmin:cmax]

        return cropped_image_np, cropped_mask_np

    def crop_image_and_mask(self, image: Image.Image, mask: Image.Image, padding: int = 20) -> tuple[Image.Image, Image.Image]:
        """根据mask同时裁剪图像和mask，避免重复计算边界框"""
        mask_np = np.array(mask)
        image_np = np.array(image)

        rows = np.any(mask_np > 0, axis=1)
        cols = np.any(mask_np > 0, axis=0)

        if not np.any(rows) or not np.any(cols):
            return image, mask

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        h, w = image_np.shape[:2]
        rmin = max(0, rmin - padding)
        rmax = min(h, rmax + padding + 1)
        cmin = max(0, cmin - padding)
        cmax = min(w, cmax + padding + 1)

        cropped_image_np = image_np[rmin:rmax, cmin:cmax]
        cropped_mask_np = mask_np[rmin:rmax, cmin:cmax]

        return Image.fromarray(cropped_image_np), Image.fromarray(cropped_mask_np)

    def _apply_mask_to_image_np(self, image_np: np.ndarray, mask_np: np.ndarray, background_color: tuple) -> np.ndarray:
        """将NumPy mask应用到NumPy图像上"""
        mask_normalized = mask_np.astype(np.float32) / 255.0
        background = np.full_like(image_np, background_color)
        result = image_np * mask_normalized[..., np.newaxis] + background * (1 - mask_normalized[..., np.newaxis])
        return result.astype(np.uint8)

    def apply_mask_to_image(self, image: Image.Image, mask: Image.Image, background_color: tuple = (255, 255, 255)) -> Image.Image:
        """
        将mask应用到图像上，mask外的区域设置为指定背景色
        
        Args:
            image: 原始图像
            mask: 二值mask图像
            background_color: 背景颜色 (R, G, B)
        
        Returns:
            应用mask后的图像
        """
        # 转换为numpy数组
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        # 将mask归一化到0-1范围
        mask_normalized = mask_np.astype(np.float32) / 255.0
        
        # 创建背景
        background = np.full_like(image_np, background_color)
        
        # 应用mask：mask区域保持原图，其他区域为背景色
        result = image_np * mask_normalized[..., np.newaxis] + background * (1 - mask_normalized[..., np.newaxis])
        
        return Image.fromarray(result.astype(np.uint8))

    def extract_head(
        self,
        image: Image.Image,
        crop_padding: int = 10,
        background_color: tuple = (255, 255, 255),
        pad2square: bool = True,
        output_mode: str = 'RGB',
        long_edge: int = 1024,
        include_shoulders: bool = False
    ) -> Image.Image:
        """
        从输入图像中提取头部区域，并返回一个裁剪、填充为正方形的图像。

        Args:
            image: 输入图像 (PIL.Image or np.ndarray).
            crop_padding: 裁剪边界框的额外边距.
            background_color: `output_mode` 为 'RGB' 时，用于填充背景的颜色.
            pad2square (bool): 是否将最终结果填充为正方形. 默认为 True.
            output_mode (str): 输出图像模式，可选 'RGB' (纯色背景) 或 'RGBA' (透明背景). 默认为 'RGB'.
            long_edge (int): 送入模型前缩放的长边尺寸，值越小速度越快，但可能影响精度。默认为1024。

        Returns:
            处理后的头部图像 (PIL.Image).
        """
        # 1. 任务类型改为可选
        task = TaskType.head_plus_shoulders if include_shoulders else TaskType.head
        processed_image_np, head_mask_np = self.process(
            image=image,
            task_type=task,
            long_edge=long_edge
        )

        # 2. NumPy-based 裁剪
        face_cropped_np, mask_cropped_np = self._crop_image_and_mask_np(
            processed_image_np, head_mask_np, padding=crop_padding
        )

        # 3. 根据输出模式（RGB/RGBA）应用蒙版
        output_mode = output_mode.upper()
        if output_mode == 'RGBA':
            # 创建一个带透明通道的RGBA图像
            # 首先确保图像是3通道的
            if face_cropped_np.shape[2] == 4:
                face_cropped_np = face_cropped_np[:,:,:3]
            # 创建RGBA图像
            result_image_np = cv2.cvtColor(face_cropped_np, cv2.COLOR_RGB2RGBA)
            result_image_np[:, :, 3] = mask_cropped_np # 设置alpha通道
            
        elif output_mode == 'RGB':
            # NumPy-based 蒙版应用
            result_image_np = self._apply_mask_to_image_np(
                face_cropped_np,
                mask_cropped_np,
                background_color=background_color
            )
        else:
            raise ValueError("output_mode must be 'RGB' or 'RGBA'")

        # 4. 可选：NumPy-based 填充
        if pad2square:
            if output_mode == 'RGBA':
                pad_color = (255, 255, 255, 0) # 透明背景
            else:  # RGB
                pad_color = background_color
            
            final_image_np = self._pad_to_square_np(
                result_image_np,
                background_value=pad_color
            )
        else:
            final_image_np = result_image_np

        # 5. 仅在最后一步转换为 PIL Image
        if output_mode == 'RGBA':
             return Image.fromarray(final_image_np, 'RGBA')
        else:
             return Image.fromarray(final_image_np, 'RGB')


    def extract(
        self,
        task_type: TaskType.full_character,
        image: Image.Image,
        crop_padding: int = 10,
        background_color: tuple = (255, 255, 255),
        pad2square: bool = True,
        output_mode: str = 'RGB',
        long_edge: int = 1024
    ) -> Image.Image:
        """
        从输入图像中提取头部区域，并返回一个裁剪、填充为正方形的图像。

        Args:
            image: 输入图像 (PIL.Image or np.ndarray).
            crop_padding: 裁剪边界框的额外边距.
            background_color: `output_mode` 为 'RGB' 时，用于填充背景的颜色.
            pad2square (bool): 是否将最终结果填充为正方形. 默认为 True.
            output_mode (str): 输出图像模式，可选 'RGB' (纯色背景) 或 'RGBA' (透明背景). 默认为 'RGB'.
            long_edge (int): 送入模型前缩放的长边尺寸，值越小速度越快，但可能影响精度。默认为1024。

        Returns:
            处理后的头部图像 (PIL.Image).
        """
        # 1. 运行分割，直接获取 NumPy 结果
        processed_image_np, head_mask_np = self.process(
            image=image,
            task_type=task_type,
            long_edge=long_edge
        )

        # 2. NumPy-based 裁剪
        face_cropped_np, mask_cropped_np = self._crop_image_and_mask_np(
            processed_image_np, head_mask_np, padding=crop_padding
        )

        # 3. 根据输出模式（RGB/RGBA）应用蒙版
        output_mode = output_mode.upper()
        if output_mode == 'RGBA':
            # 创建一个带透明通道的RGBA图像
            # 首先确保图像是3通道的
            if face_cropped_np.shape[2] == 4:
                face_cropped_np = face_cropped_np[:,:,:3]
            # 创建RGBA图像
            result_image_np = cv2.cvtColor(face_cropped_np, cv2.COLOR_RGB2RGBA)
            result_image_np[:, :, 3] = mask_cropped_np # 设置alpha通道
            
        elif output_mode == 'RGB':
            # NumPy-based 蒙版应用
            result_image_np = self._apply_mask_to_image_np(
                face_cropped_np,
                mask_cropped_np,
                background_color=background_color
            )
        else:
            raise ValueError("output_mode must be 'RGB' or 'RGBA'")

        # 4. 可选：NumPy-based 填充
        if pad2square:
            if output_mode == 'RGBA':
                pad_color = (255, 255, 255, 0) # 透明背景
            else:  # RGB
                pad_color = background_color
            
            final_image_np = self._pad_to_square_np(
                result_image_np,
                background_value=pad_color
            )
        else:
            final_image_np = result_image_np

        # 5. 仅在最后一步转换为 PIL Image
        if output_mode == 'RGBA':
             return Image.fromarray(final_image_np, 'RGBA')
        else:
             return Image.fromarray(final_image_np, 'RGB')

if __name__ == '__main__':
    # 这是一个示例如何初始化和使用 Pipeline
    print("Initializing pipeline from package resources...")
    pipeline = ProcessorPipeline.load()
    print("Pipeline initialized.")

    # 使用示例 (需要提供一张图片):

    # 请替换为你的图片路径
    image_path = "001.jpg" 
    if os.path.exists(image_path):
        print(f"Processing image: {image_path}")
        image = Image.open(image_path)
        
        print("正在提取头部...")
        extracted_head = pipeline.extract_head(image)

        # 保存最终结果
        output_path = "output_head_extracted.png"
        extracted_head.save(output_path)

        print("\n处理完成!")
        print(f"已保存提取的头部图像至 '{output_path}'")

    else:
        print(f"示例图片未找到: {image_path}")

