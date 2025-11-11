#!/bin/bash
# 这是一个用于一键式安装 head_extractor 包及其本地依赖 mmcv 的脚本。

# 如果任何命令失败，则立即退出
set -e

echo "--- 步骤 1: 安装编译和加速工具 ---"
pip install -U pip setuptools wheel ninja cmake yapf


echo "--- 步骤 2: 设置编译加速环境变量 ---"
export USE_NINJA=1
export MAX_JOBS=$(nproc)
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)


echo "--- 步骤 3: 清理旧版本，确保干净的安装环境 ---"
pip uninstall head_extractor mmcv -y || echo "旧版本未找到，继续..."
rm -rf build dist *.egg-info mmcv-2.1.0/build mmcv-2.1.0/*.egg-info


echo "--- 步骤 4: 安装 head_extractor  ---"
pip install -v .

echo "--- 步骤 5: 安装 mmcv  ---"
cd ./mmcv-2.1.0
MMCV_WITH_OPS=1 pip install -v .
cd ../


echo "--- 步骤 6: 验证安装 ---"
python -c "import head_extractor" && echo "✅ head_extractor"
python -c "import mmengine" && echo "✅ mmengine"
python -c "from mmdet.models.dense_heads import Mask2FormerHead" && echo "✅ mmdet (core)"
python -c "import mmseg" && echo "✅ mmseg"
python -c "import mmcv; import mmcv._ext" && echo "✅ mmcv"
python -c "import mmcv; print('✅ MMCV 版本:', mmcv.__version__)"

echo "🎉 全部安装和验证完成！"