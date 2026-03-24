"""日志工具"""

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from typing import Optional
from pathlib import Path

from hello_agents.utils.env_utils import get_project_path


def setup_logger(
    name: str = "hello_agents",
    level: str = "INFO",
    format_string: Optional[str] = None,
    log_dir: str = "logs",
    log_filename: Optional[str] = None
) -> logging.Logger:
    """
    设置日志记录器，支持按天生成新的日志文件

    Args:
        name: 日志记录器名称
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: 日志格式字符串
        log_dir: 日志文件存储目录
        log_filename: 日志文件名（不含扩展名），默认使用 logger 名称

    Returns:
        配置好的日志记录器
    """
    # 创建日志目录
    log_path = get_project_path() + '/logs'
    log_path.mkdir(parents=True, exist_ok=True)

    # 设置日志文件名
    if log_filename is None:
        log_filename = name

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # 避免重复添加 handler
    if not logger.handlers:
        # 设置日志格式
        formatter = logging.Formatter(
            format_string or '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # 文件处理器：每天生成一个新的日志文件
        file_handler = TimedRotatingFileHandler(
            filename=log_path / f"{log_filename}.log",
            when='midnight',  # 每天午夜切换
            interval=1,  # 间隔为1天
            backupCount=30,  # 保留30天的日志文件
            encoding='utf-8'
        )
        file_handler.suffix = "%Y-%m-%d.log"  # 设置日志文件名后缀格式
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 控制台处理器：同时输出到控制台
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_logger(name: str = "hello_agents") -> logging.Logger:
    """获取日志记录器"""

    return logging.getLogger(name)


logger = get_logger()
