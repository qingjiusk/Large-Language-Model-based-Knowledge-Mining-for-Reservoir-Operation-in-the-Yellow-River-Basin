"""
PP-UIE 本地推理客户端
使用本地 models/PP-UIE-1.5B 模型替代 DeepSeek API 进行三元组抽取
接口兼容 DeepSeekClient，可直接替换给 TripletExtractor 使用
"""
import json
import time
from typing import Any, Optional

from src.common.logger import get_logger

logger = get_logger(__name__)


class UIEClient:
    """PP-UIE 本地推理客户端，接口兼容 DeepSeekClient"""

    def __init__(
        self,
        model_path: str = "models/PP-UIE-1.5B",
        device: str = "gpu",
        precision: str = "float16",
        max_length: int = 4096,
        temperature: float = 0.0,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        # 以下参数为兼容 DeepSeekClient 构造签名，不使用但接受
        api_key: str = "",
        base_url: str = "",
        model: str = "",
        max_tokens: int = 4096,
    ):
        """
        初始化 PP-UIE 客户端

        Args:
            model_path: 模型目录路径
            device: 推理设备 ("gpu" / "cpu")
            precision: 模型精度 ("float16" / "bfloat16" / "float32")
            max_length: 最大生成长度 (tokens)
            temperature: 生成温度 (0=确定性输出)
            max_retries: JSON 解析失败时的最大重试次数
            retry_delay: 重试间隔 (秒)
            api_key, base_url, model, max_tokens: 兼容 DeepSeekClient 签名，忽略
        """
        self.model_path = model_path
        self.device = device
        self.precision = precision
        self.max_length = max_length
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._model = None
        self._tokenizer = None
        self._load_model()

        logger.info(
            f"UIEClient 初始化: model={model_path}, device={device}, "
            f"precision={precision}, max_length={max_length}"
        )

    def _load_model(self):
        """加载 PP-UIE 模型和 tokenizer"""
        # ---- monkey-patch: aistudio-sdk 0.3.8 缺少 paddlenlp 3.0.0b3 需要的 download 函数 ----
        # 使用本地模型无需远程下载功能，打桩即可
        import aistudio_sdk.hub as _hub
        if not hasattr(_hub, "download"):
            def _dummy_download(*args, **kwargs):
                raise RuntimeError(
                    "aistudio download not available, use local model instead"
                )
            _hub.download = _dummy_download
        # ----------------------------------------------------------------------------------

        from paddlenlp.transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"加载 PP-UIE 模型: {self.model_path}")
        t0 = time.time()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side="left",
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=self.precision,
        )
        self._model.eval()

        elapsed = time.time() - t0
        logger.info(f"PP-UIE 模型加载完成，耗时 {elapsed:.1f}s")

    def extract_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Any:
        """
        发送 prompt 到 PP-UIE 并返回解析后的 JSON

        Args:
            prompt: 用户 prompt
            system_prompt: 可选系统角色设定

        Returns:
            解析后的 JSON 对象 (dict/list)，失败时返回 {} 或 []
        """
        effective_system = system_prompt or "You must respond with valid JSON."

        # 构建 chat messages
        messages = [
            {"role": "system", "content": effective_system},
            {"role": "user", "content": prompt},
        ]

        # 用 Qwen chat template 格式化
        # add_generation_prompt=True 会在末尾添加 "<|im_start|>assistant\n"
        formatted_prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize — 返回 PaddlePaddle tensors
        input_features = self._tokenizer(
            formatted_prompt,
            max_length=6144,  # 留 ~2000 tokens 给生成输出
            truncation=True,
            padding_side="right",  # 右侧截断保留 prompt 开头
            return_tensors="pd",
        )

        for attempt in range(self.max_retries):
            try:
                # 生成
                from paddlenlp.generation import GenerationConfig

                gen_config = GenerationConfig.from_pretrained(self.model_path)
                gen_config.max_new_tokens = self.max_length
                gen_config.decode_strategy = (
                    "greedy_search" if self.temperature == 0 else "sampling"
                )
                gen_config.temperature = self.temperature if self.temperature > 0 else None
                gen_config.top_p = 0.8 if self.temperature > 0 else None
                gen_config.top_k = 20 if self.temperature > 0 else None
                gen_config.eos_token_id = [151645, 151643]
                gen_config.pad_token_id = 151643

                # PaddleNLP generate() 返回 tuple (token_ids, scores)
                output = self._model.generate(
                    **input_features,
                    generation_config=gen_config,
                )
                # token_ids: Tensor shape [batch_size, generated_seq_len]
                token_ids = output[0]

                # 解码 — 仅包含新生成的 tokens，无需切掉输入
                if hasattr(token_ids, "tolist"):
                    token_ids = token_ids.tolist()
                generated_ids = token_ids[0]  # 第一批
                content = self._tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                )

                logger.debug(
                    f"PP-UIE 响应 (attempt {attempt + 1}): {content[:300]}..."
                )

                # 尝试解析 JSON
                return self._parse_json_response(content)

            except json.JSONDecodeError as e:
                logger.warning(
                    f"JSON 解析失败 (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

            except Exception as e:
                logger.warning(
                    f"推理失败 (attempt {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        # 全部重试失败 — 返回空结果
        logger.error("PP-UIE JSON 提取全部重试失败，返回空结果")
        return {}

    def _parse_json_response(self, content: str) -> Any:
        """解析模型输出的 JSON，失败时尝试修复截断"""
        content = content.strip()
        if not content:
            return {}

        # 移除可能的 markdown 代码块包裹
        if content.startswith("```"):
            lines = content.split("\n")
            # 去掉首行 ```json 和末行 ```
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            content = "\n".join(lines).strip()

        # 尝试直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 尝试修复截断的 JSON
        return self._extract_json_from_text(content)

    def _extract_json_from_text(self, text: str) -> Any:
        """
        从混合文本中提取 JSON，支持截断 JSON 的回收
        （复用 DeepSeekClient 的同名逻辑）
        """
        if not text:
            return {}
        text = text.strip()
        if not text:
            return {}

        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 查找 JSON 数组或对象的开始
        start_idx = text.find("[")
        if start_idx == -1:
            start_idx = text.find("{")

        if start_idx != -1:
            candidate = text[start_idx:]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as e:
                if hasattr(e, 'pos') and e.pos > 0:
                    truncated = candidate[:e.pos]
                    for cut in range(len(truncated), 0, -1):
                        attempt = truncated[:cut].rstrip()
                        if attempt.endswith((",", ":", "{")):
                            continue
                        try:
                            fixed = self._repair_truncated_json(attempt)
                            result = json.loads(fixed)
                            if isinstance(result, (list, dict)):
                                logger.warning(
                                    f"截断 JSON 回收: "
                                    f"{len(result) if isinstance(result, list) else 'dict'} 条"
                                )
                                return result
                        except (json.JSONDecodeError, ValueError):
                            continue
                        break

        logger.warning(f"无法从文本中提取 JSON: {text[:300]}...")
        return {}

    @staticmethod
    def _repair_truncated_json(text: str) -> str:
        """尝试补全截断的 JSON 字符串"""
        # 统计未闭合的括号
        open_braces = text.count("{") - text.count("}")
        open_brackets = text.count("[") - text.count("]")

        # 检查是否在字符串中间截断
        in_string = False
        for i, c in enumerate(text):
            if c == '"' and (i == 0 or text[i - 1] != '\\'):
                in_string = not in_string

        if in_string:
            text += '"'

        text += "}" * open_braces
        text += "]" * open_brackets
        return text
