import base64
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Tuple, List, Dict, Deque

from langchain_openai import ChatOpenAI
from minio import Minio
from minio.deleteobjects import DeleteObject

from processor.config_plus.minio_config import minio_config
from processor.config_plus.vlm_config import vlm_config


from processor.import_processor.BaseNode import BaseNode
from processor.import_processor.exceptions import StateFieldError, FileProcessingError
from processor.import_processor.state import ImportGraphState
from utils.logger import logger
from utils.minio_util import get_minio_client


class NodeMDImg(BaseNode):
    """
    Markdown 图片处理节点。

    核心职责：
    1. 找出 Markdown 中实际引用的本地图片；
    2. 调用vlm，为图片生成文本摘要；
    3. 将图片上传到 MinIO；
    4. 将 Markdown 中的本地图片路径替换为 MinIO URL；
    5. 将图片摘要写入 Markdown 图片 alt 文本，使图片内容能参与后续 chunk 和向量检索。

    处理后的 Markdown 示例：
        ![图片摘要](http://minio/knowledge-base/upload-images/doc/image.png)
    """

    name = "node_md_img"

    def process(self, state: ImportGraphState):
        """
        节点主入口。

        输入 state：
            - md_path: Markdown 文件路径
            - md_content: Markdown 文件内容

        输出 state：
            - md_path: 处理后的新 Markdown 文件路径
            - md_content: 替换图片链接后的 Markdown 内容
        """

        #### 步骤一：校验md路径和md内容，返回(md内容、md路径对象、md同级images目录路径对象)
        md_content, md_path_obj, images_dir = self._step_1_get_content(state)
        # 没有 images 目录说明当前 Markdown 没有本地图片资源，直接跳过
        if not images_dir.exists():
            logger.info("无图片文件夹，跳过图片处理")
            return state

        #### 步骤二：扫描 md文件同级images 目录，筛选出md实际引用的图片，返回list[(图片文件名, 图片完整路径, 图片部分上下文)...]

        target_images = self._step_2_scan_images(md_content, images_dir)

        if not target_images:
            logger.info("未检测到 MD 中引用图片，跳过图片处理")
            return state

        #### 步骤三：调用vlm生成图片摘要，用于后续文本检索，返回结果字典{"image.png": "图片摘要文本"...}
        summaries = self._step_3_generate_summaries(md_path_obj.stem, target_images)

        #### 步骤四：上传图片到 MinIO，并把 Markdown 中的本地图片引用替换成远程 URL
        new_md_content = self._step_4_upload_and_replace(
            md_path_obj.stem,
            target_images,
            summaries,
            md_content
        )

        #### 步骤五：保存为新 Markdown 文件，避免覆盖原始文件
        new_md_file_name = self._step_5_backup_new_md_file(
            state["md_path"],
            new_md_content
        )

        # 更新工作流状态，后续节点基于处理后的 Markdown 继续执行
        state["md_content"] = new_md_content
        state["md_path"] = new_md_file_name

        return state




    def _step_1_get_content(self, state: ImportGraphState) -> Tuple[str, Path, Path]:
        """
        输入：md_path、md_content
        输出：一个元组 (md_content, md_path_obj, images_dir)
            - md_content: md原始内容
            - md_path_obj: md文件路径对象
            - images_dir: md同级 images 目录路径

        """
        # 获取md_content
        md_content = state["md_content"]

        # 获取md_path_obj
        md_path = state.get("md_path")
        # 判断路径是否为空
        if not md_path:
            raise StateFieldError(field_name="md_path", expected_type=str)

        md_path_obj = Path(md_path)
        # 判断路径对应文件是否存在
        if not md_path_obj.exists():
            raise FileProcessingError(message=f"MD 文件不存在：{md_path_obj.name}")
        # 判断路径对应文件是否为文件
        if not md_path_obj.is_file():
            raise FileProcessingError(message=f"MD 路径不是文件：{md_path_obj}")

        # 获取md_path同级目录的images目录路径
        images_dir = md_path_obj.parent / "images"


        return md_content, md_path_obj, images_dir




    def _step_2_scan_images(self, md_content: str, images_dir: Path ) -> List[Tuple[str, str, Tuple[str, str]]]:
        """
        扫描 images 目录，筛选出md正文实际引用的图片：
            1. 图片格式必须受支持config配置
            2. 图片必须被 Markdown 正文实际引用

        输入：md_content、images_dir
        返回：list[
                (图片文件名1, 图片完整路径1, 图片在md文档中的部分上下文1),
                (图片文件名2, 图片完整路径2, 图片在md文档中的部分上下文2),
                ......

            ]

        """
        # 定义一个列表用来装结果：(图片文件名, 图片完整路径, 图片上下文)
        target_images = []

        # 遍历images下的每个文件名：
        for image_file in os.listdir(images_dir):

            # ①取每一个文件名的后缀
            file_ext = os.path.splitext(image_file)[1].lower()

            # 如果当前文件名后缀不是config配置文件中支持的格式，则跳过不处理
            if file_ext not in self.config.image_extensions:
                logger.warning(f"图片格式不支持，跳过：{image_file}")
                continue

            # ②拼接出完整的图片路径
            img_path = str(images_dir / image_file)

            # ③查找该受支持的图片是否在md文件中出现(自定义方法判断md_content是否出现类似：“![描述](images/a.png)”的描述)
            #  如果出现，返回图片出现位置的部分上下文，以便后续vlm更好地总结图片信息
            context = self._find_image_in_md(md_content, image_file)
            #  如果该受支持的图片并没有在md文档中出现，则跳过不处理
            if not context:
                logger.warning(f"图片未在 MD 中引用，跳过：{image_file}")
                continue

            # 向结果列表装本次受支持、且在md文档中出现的图片的信息(图片文件名、图片路径、图片上下文)
            target_images.append((image_file, img_path, context))

        return target_images

    # 步骤二中用到的辅助方法
    def _find_image_in_md(
        self,
        md_content: str,
        image_file: str,
        context_len: int = 100
    ) -> Tuple[str, str] | None:
        """
        在 Markdown 中查找指定图片引用，并返回图片前后的文本上下文。

        上下文作用：
            图片单独看可能缺少业务背景，前后文可以帮助 VLM 生成更准确的摘要。

        匹配示例：
            ![描述](images/example.png)
        """

        # re.escape 用于处理文件名中的特殊字符，例如括号、加号、点号等
        pattern = re.compile(
            r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)"
        )

        match = pattern.search(md_content)
        if not match:
            return None

        start, end = match.span()

        # 截取图片引用前后一定长度文本，注意处理边界，避免索引越界
        pre_text = md_content[max(0, start - context_len):start]
        post_text = md_content[end:min(len(md_content), end + context_len)]

        return pre_text, post_text




    def _step_3_generate_summaries(
        self,
        doc_stem: str,
        target_images: List[Tuple[str, str, Tuple[str, str]]]
    ) -> Dict[str, str]:
        """
        为每张图片生成文本摘要。

        返回结果字典：
            {
                "image.png": "图片摘要文本",
                ...
            }

        注意：
            VLM API 通常有频率限制，因此这里在每次请求前做滑动窗口限流。
        """
        # 创建结果字典保存vlm处理结果
        summaries = {}

        # 保存最近一段时间内的请求时间戳，用于控制 API 请求频率
        request_deque = deque()

        for img_file, image_path, context in target_images:
            # 控制vlm模型请求速率
            self._apply_api_rate_limit(request_deque, max_requests=30)
            # 调用vlm做图片文本总结
            summaries[img_file] = self._summarize_image(
                image_path=image_path,
                root_folder=doc_stem,
                image_content=context
            )

        return summaries

    # 步骤三辅助方法——控制vlm模型请求速率
    def _apply_api_rate_limit(
        self,
        request_times: Deque[float],
        max_requests: int,
        window_seconds: int = 30
    ) -> None:
        """
        基于请求时间戳队列实现的滑动窗口限流器：
            在 window_seconds 时间窗口内，最多允许 max_requests 次请求。

        """

        current_time = time.time()

        # 移除窗口外的历史请求，只保留当前窗口内的请求记录
        while request_times and current_time - request_times[0] >= window_seconds:
            request_times.popleft()

        # 当前窗口请求数达到上限时，等待最早一次请求过期
        if len(request_times) >= max_requests:
            sleep_duration = window_seconds - (current_time - request_times[0])

            if sleep_duration > 0:
                logger.info(
                    f"触发 API 限流：{window_seconds}s 内最多 {max_requests} 次，"
                    f"等待 {sleep_duration:.2f}s"
                )
                time.sleep(sleep_duration)

                # 等待期间可能已有请求过期，重新清理窗口
                current_time = time.time()
                while request_times and current_time - request_times[0] >= window_seconds:
                    request_times.popleft()

        request_times.append(current_time)

        logger.info(
            f"API 请求已记录，当前 {window_seconds}s 窗口内请求数：{len(request_times)}"
        )

    # 步骤三辅助方法——调用vlm做图片文本总结
    def _summarize_image(
        self,
        image_path: str,
        root_folder: str,
        image_content: Tuple[str, str]
    ) -> str:
        """
        调用视觉语言模型vlm生成图片摘要。

        输入：
            - image_path: 图片路径
            - root_folder: 文档名
            - image_content: 图片部分上下文

        返回：
            图片中文摘要。该摘要会写入 Markdown alt 文本，并参与后续 chunk 和向量化。
        """

        with open(image_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")

        try:
            vlm = ChatOpenAI(
                model=vlm_config.vl_model,
                api_key=vlm_config.api_key,
                base_url=vlm_config.base_url,
                temperature=0.1
            )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"这是《{root_folder}》文件中的一张图片。"
                                f"图片上文为：{image_content[0]}；"
                                f"图片下文为：{image_content[1]}。"
                                f"请用中文简要总结图片内容，用作 Markdown 图片标题。"
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ]

            response = vlm.invoke(messages)

            # 去掉换行，避免写入 Markdown alt 文本后影响格式
            return response.content.strip().replace("\n", "")

        except Exception as e:
            logger.error(f"图像总结失败：{image_path}，错误：{e}")

            # 失败时给默认描述，保证后续 Markdown 替换流程不中断
            return "图片描述"




    def _step_4_upload_and_replace(
        self,
        doc_stem: str,
        target_images: List[Tuple[str, str, Tuple[str, str]]],
        summaries: Dict[str, str],
        md_content: str
    ) -> str:
        """
        上传图片到 MinIO，并替换 Markdown 中的本地图片引用。

        流程：
            1. 获取 MinIO 客户端；
            2. 按文档名构造上传目录；
            3. 清理旧目录，保证重复导入时结果幂等；
            4. 批量上传图片；
            5. 合并图片摘要和 URL；
            6. 替换 Markdown 图片语法。
        """

        # 获取minio客户端
        minio_client = get_minio_client()

        # 为md图片配置共同的minio对象前缀名
        upload_dir = f"{minio_config.img_dir}/{doc_stem}".replace(" ", "")

        # 清理配置目录的旧minio文件对象
        self._clean_minio_directory(minio_client, upload_dir)

        # 批量上传图片至minio，获取对应的urls字典
        urls = self._upload_images_batch(
            minio_client=minio_client,
            upload_dir=upload_dir,
            target_images=target_images
        )

        # 合并图片摘要和minio 对象url，过滤上传失败的图片
        image_info = self._merge_summary_and_url(
            summaries=summaries,
            urls=urls
        )

        # 替换MD文档中的本地图片引用为MinIO远程引用
        return self._process_md_file(md_content, image_info)

    # 步骤四辅助方法：清理 MinIO 中指定文档目录下的旧图片。
    def _clean_minio_directory(self, minio_client: Minio, prefix: str) -> None:

        try:
            objects_to_delete = minio_client.list_objects(
                minio_config.bucket_name,
                prefix=prefix,
                recursive=True
            )

            delete_list = [
                DeleteObject(obj.object_name)
                for obj in objects_to_delete
            ]

            if delete_list:
                errors = minio_client.remove_objects(
                    minio_config.bucket_name,
                    delete_list
                )

                for error in errors:
                    logger.error(f"删除 MinIO 对象失败：{error}")

        except Exception as e:
            logger.error(f"清理 MinIO 目录失败：{e}")

    # 步骤四辅助方法：批量上传图片至minio获取对应的urls字典
    def _upload_images_batch(
        self,
        minio_client: Minio,
        upload_dir: str,
        target_images: List[Tuple]
    ) -> Dict[str, str]:
        """
        批量上传图片到 MinIO。

        返回：
            {
                "image.png": "http://minio/xxx/image.png",
                ...
            }
        """

        urls = {}

        for img_file, img_path, _ in target_images:
            object_name = f"{upload_dir}/{img_file}"

            urls[img_file] = self._upload_to_minio(
                minio_client=minio_client,
                local_path=img_path,
                object_name=object_name
            )

        return urls

    def _upload_to_minio(
        self,
        minio_client: Minio,
        local_path: str,
        object_name: str
    ) -> str | None:
        """
        上传单张图片到 MinIO，并返回可访问 URL。

        注意：
            当前 URL 拼接依赖 bucket 已配置 public-read 策略。
            如果 bucket 是私有的，这里应该改成生成预签名 URL。
        """

        try:
            file_ext = os.path.splitext(local_path)[1][1:]

            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                object_name=object_name,
                file_path=local_path,
                content_type=f"image/{file_ext}"
            )

            # 避免 Windows 路径反斜杠影响 URL 访问
            object_name = object_name.replace("\\", "%5C")

            base_url = f"http://{minio_config.endpoint}/{minio_config.bucket_name}"

            return f"{base_url}/{object_name}"

        except Exception as e:
            logger.error(
                f"图片上传 MinIO 失败：{local_path}，错误信息：{str(e)}"
            )
            return None

    # 步骤四辅助方法：合并图片摘要和minio 对象url，过滤上传失败的图片
    def _merge_summary_and_url(
        self,
        summaries: Dict[str, str],
        urls: Dict[str, str]
    ) -> Dict[str, Tuple[str, str]]:
        """
        合并图片摘要和 MinIO URL。

        只保留上传成功的图片：
            - 有摘要
            - 有 URL

        返回：
            {
                "image.png": ("图片摘要", "http://minio/xxx/image.png"),
                ...
            }
        """

        image_info = {}

        for image_file, summary in summaries.items():
            url = urls.get(image_file)

            if url:
                image_info[image_file] = (summary, url)

        return image_info

    # 步骤四辅助方法：替换md文档中的图片引用格式为：![图片摘要](http://minio/xxx/image.png)
    def _process_md_file(
        self,
        md_content: str,
        image_info: Dict[str, Tuple[str, str]]
    ) -> str:
        """
        替换 Markdown 中的图片引用。

        替换前：
            ![旧描述](images/image.png)

        替换后：
            ![图片摘要](http://minio/xxx/image.png)

        说明：
            图片摘要用于文本检索；
            MinIO URL 用于前端或知识库页面展示原图。
        """

        for image_file, (summary, new_url) in image_info.items():
            pattern = re.compile(
                r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)"
            )

            # 使用 lambda 返回替换字符串，避免 summary 中的反斜杠被正则替换引擎误解析
            md_content = pattern.sub(
                lambda _: f"![{summary}]({new_url})",
                md_content
            )

        logger.info(
            f"MD 图片引用替换完成，共替换 {len(image_info)} 张图片"
        )

        return md_content





    def _step_5_backup_new_md_file(
        self,
        origin_md_path: str,
        md_content: str
    ) -> str:
        """
        将处理后的 Markdown 保存为新文件。

        命名规则：
            xxx.md -> xxx_new.md

        这样可以保留原始解析结果，方便排查图片替换或 VLM 摘要问题。
        """

        new_md_file_name = os.path.splitext(origin_md_path)[0] + "_new.md"

        with open(new_md_file_name, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"处理后 MD 文件已保存：{new_md_file_name}")

        return new_md_file_name