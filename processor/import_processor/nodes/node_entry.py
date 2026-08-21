from pathlib import Path
from processor.import_processor.BaseNode import BaseNode
from processor.import_processor.exceptions import StateFieldError, ValidationError
from processor.import_processor.state import ImportGraphState
from utils.logger import logger


class NodeEntry(BaseNode):


    # 当前节点名称
    name = "node_entry"

    def process(self, state: ImportGraphState):
        """
        任务分发节点

        内容：
            1. 校验用户上传的文件路径state[import_file_path]是否有效
            2. 判断文件类型（PDF / Markdown），更新state对应类型的标记和路径字段
            3. 提取文件名更新到state[file_title]

        Args:
            state[import_file_path]

        Returns:
            state[is_pdf_read_enabled],state[md_path] / state[is_md_read_enabled],state[md_path]
            state[file_title]

        """

        # 获取上传文件路径
        import_file_path = state.get('import_file_path', '')

        # 若路径为空抛出异常
        if not import_file_path:
            raise StateFieldError(
                node_name=self.name,
                field_name='import_file_path',
                expected_type=str
            )

        import_file_path_obj = Path(import_file_path)

        # 若文件不存在抛出异常
        if not import_file_path_obj.exists():
            raise StateFieldError(
                node_name=self.name,
                field_name='import_file_path_obj',
                expected_type=Path
            )

        # 判断文件类型，设置state字段
        if import_file_path_obj.suffix == '.pdf':

            state['is_pdf_read_enabled'] = True
            state['pdf_path'] = import_file_path

        elif import_file_path_obj.suffix == '.md':

            state['is_md_read_enabled'] = True
            state['md_path'] = import_file_path
            # 直接将md文件内容更新到state[md_content]
            state["md_content"] = import_file_path_obj.read_text(encoding="utf-8-sig")

        else:
            logger.error(
                f"该文件后缀格式 {import_file_path_obj.suffix} 不支持"
            )

            raise ValidationError(
                message=f"该文件的后缀格式 {import_file_path_obj.suffix} 不支持",
                node_name=self.name
            )

        # 提取文件名
        state['file_title'] = import_file_path_obj.stem

        # 更新状态
        return state


