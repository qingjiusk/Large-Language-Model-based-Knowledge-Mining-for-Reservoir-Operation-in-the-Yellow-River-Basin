"""
配置加载器
基于 YAML 配置文件，支持点号分隔路径访问和环境变量替换
"""
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv


class ConfigLoader:
    """YAML 配置加载器，支持环境变量插值和点号分隔的 key 访问"""

    def __init__(self, config_path: str = "config/config.yaml"):
        # 加载 .env 文件
        load_dotenv()

        self.config_path = Path(config_path)
        self._raw_config = self._load_raw()
        self._config = self._resolve_env_vars(self._raw_config)

    def _load_raw(self) -> Dict:
        """读取原始 YAML 内容"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _resolve_env_vars(self, config: Any) -> Any:
        """
        递归解析配置中的环境变量占位符
        支持两种格式:
          - ${VAR_NAME}          直接使用环境变量
          - ${VAR_NAME:default}  使用环境变量，不存在时用默认值
        """
        pattern = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")

        if isinstance(config, dict):
            return {k: self._resolve_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._resolve_env_vars(item) for item in config]
        elif isinstance(config, str):
            def replacer(match):
                var_name = match.group(1)
                default = match.group(2)
                return os.environ.get(var_name, default if default is not None else match.group(0))
            return pattern.sub(replacer, config)
        else:
            return config

    def get(self, key: str, default: Any = None) -> Any:
        """
        使用点号分隔的 key 访问嵌套配置
        例如: cfg.get("deepseek.model") 返回 config["deepseek"]["model"]
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value

    def get_deepseek_config(self) -> Dict:
        """获取 DeepSeek API 完整配置"""
        return self._config.get("deepseek", {})

    def get_neo4j_config(self) -> Dict:
        """获取 Neo4j 连接完整配置"""
        return self._config.get("neo4j", {})

    def get_data_paths(self) -> Dict:
        """获取数据路径配置"""
        return self._config.get("data", {})

    def to_dict(self) -> Dict:
        """返回完整配置字典"""
        return self._config

    def __repr__(self) -> str:
        return f"ConfigLoader(config_path='{self.config_path}')"
