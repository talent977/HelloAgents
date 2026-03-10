import configparser
import os
import platform


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