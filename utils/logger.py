"""
    配置日志logger
"""
import logging

import colorlog

# 创建logger
logger = logging.getLogger()
# 配置logger过滤等级
logger.setLevel(logging.INFO)

# 打印到控制台设置
handler = colorlog.StreamHandler()
# 打印到控制台格式
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
))

logger.handlers.clear()
logger.addHandler(handler)