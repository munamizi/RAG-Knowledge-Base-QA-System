"""
查询流程节点基类
"""
from abc import abstractmethod, ABC
from typing import TypeVar

from utils.logger import logger
from utils.task_util import add_running_task, add_done_task

T = TypeVar("T")  # 泛型状态类型
class BaseNode(ABC):
    # 节点名称，子类应覆盖
    name: str = "base_node"


    @abstractmethod
    def process(self, state: T) -> T:
        """
        子类业务节点必须实现此方法
        """
        pass



    def __call__(self, state: T) -> T:
        """
        节点执行入口
        """
        try:

            logger.info(f"--- {self.name} 开始 ---")

            add_running_task(state['session_id'], self.name, state.get("is_stream"))

            result = self.process(state)

            add_done_task(state.get("session_id"), self.name, state.get("is_stream"))

            logger.info(f"--- {self.name} 完成 ---")

            return result

        except Exception as e:
            logger.error(f"{self.name} 执行失败: {e}")
            raise

