from pathlib import Path
from processor.import_processor.BaseNode import BaseNode
from processor.import_processor.exceptions import StateFieldError, ValidationError
from processor.import_processor.state import ImportGraphState
from utils.logger import logger


class NodeEntry(BaseNode):
    """
    入口节点：“导入流程的任务分发器”

        作用：
            1. 校验用户上传的文件路径是否合法
            2. 判断文件类型（PDF / Markdown）
            3. 为后续条件边设置标记
            4. 提取文件标题作为元数据

    """

    # 当前节点名称
    name = "node_entry"

    def process(self, state: ImportGraphState):

        # 1. 获取上传文件路径
        import_file_path = state.get('import_file_path', '')

        # 若路径为空抛出异常
        if not import_file_path:
            raise StateFieldError(
                node_name=self.name,
                field_name='import_file_path',
                expected_type=str
            )

        # 2. 路径Path对象化方便操作路径
        import_file_path_obj = Path(import_file_path)

        # 若文件不存在抛出异常
        if not import_file_path_obj.exists():
            raise StateFieldError(
                node_name=self.name,
                field_name='import_file_path_obj',
                expected_type=Path
            )

        # 3. 根据文件类型设置条件边的标记
        if import_file_path_obj.suffix == '.pdf':

            state['is_pdf_read_enabled'] = True
            state['pdf_path'] = import_file_path

        elif import_file_path_obj.suffix == '.md':

            state['is_md_read_enabled'] = True
            state['md_path'] = import_file_path

        else:
            logger.error(
                f"该文件后缀格式 {import_file_path_obj.suffix} 不支持"
            )

            raise ValidationError(
                message=f"该文件的后缀格式 {import_file_path_obj.suffix} 不支持",
                node_name=self.name
            )

        # 4. 提取文件名
        state['file_title'] = import_file_path_obj.stem

        # 5. 返回更新后的状态
        return state


