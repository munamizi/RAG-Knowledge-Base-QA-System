from processor.import_processor.exceptions import ImportProcessError
from processor.import_processor.state import ImportGraphState
from abc import ABC, abstractmethod

from utils.logger import logger
from utils.task_util import add_running_task, add_done_task


class BaseNode(ABC):
    """
    基类节点
        1.提供所有节点必统一的日志和错误处理。
        2.所有子类节点继承此基类，实现 process 方法。
    """

    name: str = "base_node"  # 节点名称,子类覆盖

    @abstractmethod
    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        子类业务节点必须实现此方法
        """
        pass



    def __call__(self, state: ImportGraphState) -> ImportGraphState:
        """
        LangGraph 调用节点时会调用此方法。
        """
        try:
            # 记录日志开始
            logger.info(f"--- {self.name} 开始 ---")
            # 记录节点开始状态
            add_running_task(state["task_id"], self.name)

            # 节点执行
            result = self.process(state)

            # 记录节点结束状态
            add_done_task(state["task_id"], self.name)

            # 记录日志结束
            logger.info(f"--- {self.name} 完成 ---")

            return result
        except Exception as e:
            logger.error(f"{self.name} 执行失败: {e}")
            raise ImportProcessError(
                message=str(e),
                node_name=self.name,
                cause=e
            )


