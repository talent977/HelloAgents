#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 环境变量读取
import configparser
import os
import platform
from pathlib import Path
from urllib.parse import urlparse, urlunparse


def is_linux():
    os = platform.system()
    if os == 'Linux':
        return True
    else:
        return False


def get_config_path():
    if is_linux():
        # todo linux 默认读取本机配置文件！！

        ini_abs_path = '/opt/zy/python_service/.acm-config.ini'
    else:
        ini_abs_path = os.path.join(os.path.abspath(os.path.dirname(__file__))) + '/../conf/config.ini'
    return ini_abs_path


def get_env_info(file_path: str = get_config_path()):
    """
    获取环境变量信息
    :param file_path:
    :return:
    """
    config_parser = configparser.ConfigParser()
    config_parser.read(file_path, encoding="utf-8")
    return config_parser


def get_project_path():
    """获取项目路径"""
    return Path(__file__).resolve().parent.parent.parent


def get_config_info(schema: str, key: str):
    """获取配置文件信息"""
    config_path = get_config_path()
    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")

    if not config.has_section(schema):
        raise ValueError(f"配置文件中未找到 section: [{schema}]")
    if not config.has_option(schema, key):
        raise ValueError(f"section [{schema}] 中未找到 key: {key}")

    return config.get(schema, key)


def get_config_section(schema: str) -> dict:
    """
    获取 config.ini 中指定 schema（section）下的所有配置项，以字典形式返回。

    Args:
        schema (str): 配置文件中的 section 名称，例如 "deepseek-ai/DeepSeek-V3.2"

    Returns:
        dict: {key: value} 形式的配置字典

    Raises:
        FileNotFoundError: 配置文件不存在
        ValueError: 指定的 schema 不存在
    """
    config_path = get_config_path()

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")

    config = configparser.ConfigParser()
    config.read(config_path, encoding="utf-8")

    if not config.has_section(schema):
        raise ValueError(f"配置文件中未找到 section: [{schema}]")

    # 将 ConfigParser 的 section 转为普通字典
    return dict(config[schema])


def get_llm_url(url: str) -> str:
    """
    根据系统环境替换 URL 中的域名部分。
    Linux 系统使用内网域名，非 Linux 系统保持原 URL 不变。

    Args:
        url: 原始 URL，例如 'https://dashscope.aliyuncs.com/compatible-mode/v1'

    Returns:
        替换域名后的 URL
    """
    if is_linux():
        # Linux 系统：替换域名为内网地址
        parsed = urlparse(url)
        # 替换域名为内网域名，保留路径和查询参数
        new_url = urlunparse(parsed._replace(netloc='snat.msuncloud-internal.com'))
        return new_url
    else:
        # 非 Linux 系统：保持原 URL 不变
        return url


if __name__ == '__main__':
    value = get_config_info('qwen3-max', 'base_url')
    print(value)
    schema = get_config_section('qwen3-max')
    print(schema)