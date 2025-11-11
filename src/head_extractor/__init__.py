from .processor import ProcessorPipeline, TaskType

__version__ = "0.1.0"

# 让外部可以直接 from head_extractor import ProcessorPipeline
__all__ = ['ProcessorPipeline', 'TaskType']