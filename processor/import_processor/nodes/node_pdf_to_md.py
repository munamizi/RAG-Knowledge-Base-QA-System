import shutil
import time
import zipfile
from pathlib import Path
import requests

from processor.config_plus.mineru_config import mineru_config
from processor.import_processor.BaseNode import BaseNode
from processor.import_processor.exceptions import (
    StateFieldError,
    FileProcessingError,
    ConfigurationError,
    PdfConversionError,
)
from processor.import_processor.state import ImportGraphState
from utils.logger import logger


class NodePDFToMD(BaseNode):
    """
    PDF转Markdown 节点（MinerU）

    内容：
        1.在state里拿pdf_path(pdf文件路径)和file_dir(转化为md文件的输出目录)，校验路径
        2.调MinerU云端API：上传本地pdf文件，轮询解析结果zip_url
        3.下载并解压md文件
        4.读取md内容
        5.更新state对应字段：md_path、md_content

    """

    name = "node_pdf_to_md"

    def process(self, state: ImportGraphState):
        """
        将pdf通过MineU转化为md文档节点

        内容：
            1.在state里拿pdf_path和 file_dir(转md后最终输出的目录)，校验
            2.调MinerU云端API：上传本地pdf文件，轮询解析结果zip_url
            3.根据zip_url下载并解压md文件，输出到对应目录
            4.读取md内容
            5.更新state对应字段：md_path、md_content

        Args:
            state[pdf_path] - 上传的pdf文件路径
            state[file_dir] - 转md后最终要输出的目录路径

        Returns:
            state[md_path] - 输出目录
            state[md_content] - 读取的md文件内容

        """

        # 1. 校验两个路径，返回对应Path对象
        pdf_path_obj, output_dir_obj = self._step_1_validate_paths(state)

        # 2. 上传 PDF 并轮询解析结果
        zip_url = self._step_2_upload_and_poll(pdf_path_obj)

        # 3. 下载并解压包为目录，返回输出目录路径
        md_path = self._step_3_download_and_extract(
            zip_url=zip_url,
            output_dir_obj=output_dir_obj,
            pdf_stem=pdf_path_obj.stem,
        )

        # 4. 读取 md 内容
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # 5. 更新对应state字段
        return {
            "md_path": str(md_path),
            "md_content": md_content,
        }



    def _step_1_validate_paths(self, state: ImportGraphState):
        """
        校验 PDF 路径与输出目录
        """

        # 获取pdf路径
        pdf_path = state.get("pdf_path")
        if not pdf_path:
            raise StateFieldError(field_name="pdf_path", expected_type=str)

        # 获取输出目录
        file_dir = state.get("file_dir")
        if not file_dir:
            raise StateFieldError(field_name="file_dir", expected_type=str)

        # 转换为 Path 对象
        pdf_path_obj = Path(pdf_path)
        file_dir_obj = Path(file_dir)

        # 校验 PDF路径对应文件是否存在
        if not pdf_path_obj.exists():
            raise FileProcessingError(message=f"PDF 文件不存在：{pdf_path_obj}")

        # 保证输出目录对应真实目录存在
        if not file_dir_obj.exists():
            logger.info(f"输出目录不存在，自动创建：{file_dir_obj.absolute()}")
            file_dir_obj.mkdir(parents=True, exist_ok=True)

        return pdf_path_obj, file_dir_obj



    def _step_2_upload_and_poll(self, pdf_path_obj: Path):
        """
        上传 PDF 到 MinerU，并轮询解析结果zip_url
        """

        # 校验 MinerU 配置
        if not mineru_config.base_url:
            raise ConfigurationError(
                "缺少 MINERU_BASE_URL 配置"
            )

        if not mineru_config.api_token:
            raise ConfigurationError(
                "缺少 MINERU_API_TOKEN 配置"
            )

        token = mineru_config.api_token

        # 获取上传链接
        url = f"{mineru_config.base_url}/file-urls/batch"
        header = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        data = {
            "files": [
                {"name": pdf_path_obj.name}
            ],
            "model_version": "vlm",
        }

        response = requests.post(
            url,
            headers=header,
            json=data,
        )

        # HTTP 校验
        if response.status_code != 200:
            raise PdfConversionError(
                message=f"获取上传链接失败：{response.status_code}"
            )

        # 业务码校验
        result = response.json()
        if result.get("code") != 0:
            raise PdfConversionError(
                f"获取上传链接失败：{result}"
            )

        # 获取上传 URL 与 batch_id
        signed_url = result["data"]["file_urls"][0]
        batch_id = result["data"]["batch_id"]

        # 3. 上传 PDF 文件
        with open(pdf_path_obj, "rb") as f:
            res_upload = requests.put(
                signed_url,
                data=f,
            )

        if res_upload.status_code != 200:
            raise PdfConversionError(
                f"PDF 上传失败：{res_upload.status_code}"
            )

        logger.info(f"PDF 上传成功：{pdf_path_obj.name}")

        # 4. 轮询解析状态
        poll_url = (
            f"{mineru_config.base_url}/extract-results/batch/{batch_id}"
        )

        start_time = time.time()

        timeout_seconds = 600
        poll_interval = 8

        while True:

            elapsed_time = time.time() - start_time

            # 超时退出
            if elapsed_time > timeout_seconds:
                raise TimeoutError(
                    f"任务超时：batch_id={batch_id}"
                )

            try:
                res_poll = requests.get(
                    url=poll_url,
                    headers=header,
                    timeout=10,
                )
            except Exception as e:   # 网络异常重试
                logger.warning(
                    f"轮询异常：{e}"
                )
                time.sleep(poll_interval)
                continue

            # HTTP 校验
            if res_poll.status_code != 200:
                raise PdfConversionError(
                    f"轮询失败：{res_poll.status_code}"
                )

            poll_data = res_poll.json()

            # 业务码校验
            if poll_data.get("code") != 0:
                raise PdfConversionError(
                    f"轮询业务失败：{poll_data}"
                )

            result_item = poll_data["data"]["extract_result"][0]
            data_state = result_item["state"]

            # 解析成功
            if data_state == "done":

                full_zip_url = result_item["full_zip_url"]

                logger.info(
                    f"解析完成：batch_id={batch_id}"
                )

                return full_zip_url

            # 解析失败
            if data_state == "failed":

                err_msg = result_item.get(
                    "err_msg",
                    "未知错误"
                )

                raise PdfConversionError(
                    f"解析失败：{err_msg}"
                )

            # 继续轮询
            logger.info(
                f"处理中：{data_state}"
            )

            time.sleep(poll_interval)



    def _step_3_download_and_extract(self,zip_url: str, output_dir_obj: Path, pdf_stem: str,) -> str:
        """下载 ZIP 并解压 MD到指定目录"""

        # 下载 ZIP(同步阻塞)到内存
        response = requests.get(zip_url)

        if response.status_code != 200:
            raise RuntimeError(
                f"ZIP 下载失败：{response.status_code}"
            )
        # 将内存zip持久化进外存
        zip_save_path = (
            output_dir_obj / f"{pdf_stem}_result.zip"
        )

        with open(zip_save_path, "wb") as f:
            f.write(response.content)

        # 准备解压目录
        extract_target_dir = output_dir_obj / pdf_stem

        # 如果存在旧目录，全部删除重新建立
        if extract_target_dir.exists():
            shutil.rmtree(extract_target_dir)
        extract_target_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # 解压 ZIP
        with zipfile.ZipFile(zip_save_path, "r") as zip_file_obj:
            zip_file_obj.extractall(extract_target_dir)

        # 重命名 md 文件
        target_md_file = extract_target_dir / "full.md"

        new_md_path = target_md_file.with_name(
            f"{pdf_stem}.md"
        )

        target_md_file.rename(new_md_path)

        return str(new_md_path.absolute())



# 单元测试
if __name__ == "__main__":
    init_state = {
        "pdf_path": r"C:\Users\Zong6\Desktop\基于大语言模型的深远海鱼类视觉检测与多模态辅助决策系统.pdf",
        "file_dir": r"C:\Users\Zong6\Desktop\MinerU转换结果",
    }

    node_pdf_to_md = NodePDFToMD()

    result = node_pdf_to_md(init_state)



