"""
DeepSeek LLM 客户端
基于 OpenAI 兼容接口，封装 DeepSeek API 调用
"""
import json
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

from src.common.logger import get_logger

logger = get_logger(__name__)


class DeepSeekClient:
    """DeepSeek API 客户端，兼容 OpenAI SDK 接口"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        """
        初始化 DeepSeek 客户端

        Args:
            api_key: DeepSeek API 密钥
            base_url: API 地址
            model: 模型名称 (deepseek-chat / deepseek-reasoner)
            max_tokens: 最大输出 token 数
            temperature: 采样温度 (0 为确定性输出)
            max_retries: 最大重试次数
            retry_delay: 重试间隔 (秒)
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        logger.info(f"DeepSeekClient 初始化: model={model}, base_url={base_url}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        通用对话接口

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            system_prompt: 可选 system prompt
            temperature: 覆盖默认 temperature
            max_tokens: 覆盖默认 max_tokens

        Returns:
            LLM 回复文本
        """
        if system_prompt:
            full_messages = [{"role": "system", "content": system_prompt}] + list(messages)
        else:
            full_messages = list(messages)

        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=full_messages,
                    temperature=temp,
                    max_tokens=tokens,
                )
                content = response.choices[0].message.content
                logger.debug(f"LLM 响应 (attempt {attempt + 1}): {content[:200]}...")
                return content

            except Exception as e:
                last_error = e
                logger.warning(f"API 调用失败 (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # 指数退避

        raise RuntimeError(f"DeepSeek API 调用失败，已重试 {self.max_retries} 次: {last_error}")

    def extract_json(self, prompt: str, system_prompt: Optional[str] = None) -> Any:
        """
        发送 prompt 并解析 JSON 响应

        Args:
            prompt: 用户 prompt
            system_prompt: 可选系统角色设定

        Returns:
            解析后的 JSON 对象 (dict/list)
        """
        messages = [{"role": "user", "content": prompt}]

        # 尝试使用 JSON mode
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=(
                        [{"role": "system", "content": system_prompt}] + messages
                        if system_prompt
                        else messages
                    ),
                    temperature=0,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                return json.loads(content)

            except json.JSONDecodeError as e:
                logger.warning(f"JSON 解析失败 (attempt {attempt + 1}): {e}")
                # 降级：尝试从非 JSON 响应中提取
                pass
            except Exception as e:
                logger.warning(f"API 调用失败 (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        # Fallback: 普通调用 + 手动提取 JSON
        logger.info("JSON mode 失败，降级为普通文本调用")
        raw_text = self.chat(messages, system_prompt=system_prompt)
        return self._extract_json_from_text(raw_text)

    def _extract_json_from_text(self, text: str) -> Any:
        """
        从混合文本中提取 JSON 数组或对象
        """
        text = text.strip()

        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 查找第一个 [ 或 { 到最后一个 ] 或 }
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            start_idx = text.find(start_char)
            end_idx = text.rfind(end_char)
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                try:
                    return json.loads(text[start_idx: end_idx + 1])
                except json.JSONDecodeError:
                    continue

        logger.error(f"无法从文本中提取 JSON: {text[:500]}")
        return [] if text.strip().startswith("[") else {}
