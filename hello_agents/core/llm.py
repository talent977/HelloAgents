"""HelloAgents统一LLM接口 - 基于OpenAI原生API"""

import os
from typing import Literal, Optional, Iterator, Any, cast
from openai import OpenAI

from .exceptions import HelloAgentsException
from ..utils.env_utils import get_config_section, get_llm_url

# 支持的LLM提供商
SUPPORTED_PROVIDERS = Literal[
    "openai",
    "deepseek",
    "qwen3-max",
    "qwen-plus",
    "modelscope",
    "kimi",
    "zhipu",
    "ollama",
    "vllm",
    "local",
    "auto",
    "custom",
]

class HelloAgentsLLM:
    """
    为HelloAgents定制的LLM客户端。
    它用于调用任何兼容OpenAI接口的服务，并默认使用流式响应。

    设计理念：
    - 参数优先，环境变量兜底
    - 流式响应为默认，提供更好的用户体验
    - 支持多种LLM提供商
    - 统一的调用接口
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: str = 'qwen3-max',
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        **kwargs
    ):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        支持自动检测provider或使用统一的LLM_*环境变量配置。

        Args:
            model: 模型名称，如果未提供则从环境变量LLM_MODEL_ID读取
            api_key: API密钥，如果未提供则从环境变量读取
            base_url: 服务地址，如果未提供则从环境变量LLM_BASE_URL读取
            provider: LLM提供商，如果未提供则自动检测
            temperature: 温度参数
            max_tokens: 最大token数
            timeout: 超时时间，从环境变量LLM_TIMEOUT读取，默认60秒
        """
        # 优先使用传入参数，如果未提供，则从环境变量加载
        if model and api_key and base_url:
            self.api_key = api_key
            self.base_url = base_url
            self.model = model
        else:
            # 自动检测provider或使用指定的provider
            requested_provider = (provider or "") if provider else None

            if requested_provider == "custom":
                self.provider = "custom"
                self.api_key = api_key or os.getenv("LLM_API_KEY")
                self.base_url = base_url or os.getenv("LLM_BASE_URL")
            else:
                # 根据provider确定API密钥和base_url
                default_llm = get_config_section(requested_provider)
                base_url = base_url or default_llm.get('base_url')
                domain_is_diff = default_llm.get('domain_is_diff')
                if domain_is_diff and domain_is_diff == 'True':
                    # 通过 get_llm_url() 替换 URL 中的域名
                    base_url = get_llm_url(base_url)
                self.base_url = base_url
                self.api_key = default_llm.get('api_key')
                self.model = default_llm.get('model_id')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.kwargs = kwargs

        # 验证必要参数
        if not all([self.api_key, self.base_url]):
            raise HelloAgentsException("API密钥和服务地址必须被提供或在.env文件中定义。")

        # 创建OpenAI客户端
        self._client = self._create_client()

    def _create_client(self) -> OpenAI:
        """创建OpenAI客户端"""
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout
        )

    def think(self, messages: list[dict[str, Any]], temperature: Optional[float] = None) -> Iterator[str]:
        """
        调用大语言模型进行思考，并返回流式响应。
        这是主要的调用方法，默认使用流式响应以获得更好的用户体验。

        Args:
            messages: 消息列表
            temperature: 温度参数，如果未提供则使用初始化时的值

        Yields:
            str: 流式响应的文本片段
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            # 使用字典构建参数，避免类型检查问题
            create_kwargs: dict[str, Any] = {
                "model": cast(str, self.model),
                "messages": messages,
                "stream": True,
            }
            if temperature is not None:
                create_kwargs["temperature"] = temperature
            elif self.temperature is not None:
                create_kwargs["temperature"] = self.temperature
            if self.max_tokens is not None:
                create_kwargs["max_tokens"] = self.max_tokens

            response = self._client.chat.completions.create(**create_kwargs)

            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if content:
                    print(content, end="", flush=True)
                    yield content
            print()  # 在流式输出结束后换行

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            raise HelloAgentsException(f"LLM调用失败: {str(e)}")

    def invoke(self, messages: list[dict[str, Any]], **kwargs) -> str:
        """
        非流式调用LLM，返回完整响应。
        适用于不需要流式输出的场景。
        """
        try:
            create_kwargs: dict[str, Any] = {
                "model": cast(str, self.model),
                "messages": messages,
            }
            # 处理temperature
            temperature = kwargs.get('temperature')
            if temperature is not None:
                create_kwargs["temperature"] = temperature
            elif self.temperature is not None:
                create_kwargs["temperature"] = self.temperature
            # 处理max_tokens
            max_tokens = kwargs.get('max_tokens', self.max_tokens)
            if max_tokens is not None:
                create_kwargs["max_tokens"] = max_tokens
            # 添加其他kwargs
            for k, v in kwargs.items():
                if k not in ['temperature', 'max_tokens']:
                    create_kwargs[k] = v

            response = self._client.chat.completions.create(**create_kwargs)
            return response.choices[0].message.content
        except Exception as e:
            raise HelloAgentsException(f"LLM调用失败: {str(e)}")

    def stream_invoke(self, messages: list[dict[str, Any]], **kwargs) -> Iterator[str]:
        """
        流式调用LLM的别名方法，与think方法功能相同。
        保持向后兼容性。
        """
        temperature = kwargs.get('temperature')
        yield from self.think(messages, temperature)
