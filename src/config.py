"""
配置管理模块
============

负责加载、合并、验证 YAML 配置文件。
支持 base.yaml + overlay（如 debug.yaml）的层级覆盖。

用法::

    from src.config import load_config
    cfg = load_config("configs/base.yaml")
    cfg = load_config("configs/base.yaml", "configs/debug.yaml")
"""

import copy
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

logger = logging.getLogger(__name__)


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并两个字典，overlay 覆盖 base 中的同名键。"""
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class Config:
    """配置容器，支持属性访问和字典访问。

    Examples::

        cfg = Config({"grid": {"nx": 128}})
        assert cfg.grid.nx == 128
        assert cfg["grid"]["nx"] == 128
    """

    def __init__(self, data: Dict[str, Any]):
        super().__setattr__("_data", {})
        for key, value in data.items():
            setattr(self, key, value)

    def __setattr__(self, key: str, value: Any) -> None:
        if key == "_data":
            super().__setattr__(key, value)
            return

        wrapped = Config(value) if isinstance(value, dict) else value
        super().__setattr__(key, wrapped)

        if "_data" in self.__dict__:
            self._data[key] = (
                wrapped.to_dict() if isinstance(wrapped, Config) else copy.deepcopy(wrapped)
            )

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def to_dict(self) -> Dict[str, Any]:
        """转换回普通字典。"""
        result = {}
        for key, value in self.__dict__.items():
            if key == "_data":
                continue
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = copy.deepcopy(value)
        return result

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


def load_config(
    base_path: Union[str, Path],
    overlay_path: Optional[Union[str, Path]] = None,
) -> Config:
    """加载配置文件。

    Args:
        base_path: 基础配置文件路径（通常为 configs/base.yaml）。
        overlay_path: 可选的覆盖配置文件路径（如 configs/debug.yaml）。

    Returns:
        Config 对象，支持属性访问。

    Raises:
        FileNotFoundError: 配置文件不存在。
        yaml.YAMLError: YAML 解析失败。
    """
    base_path = Path(base_path)
    if not base_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {base_path}")

    with open(base_path, "r", encoding="utf-8") as f:
        base_data = yaml.safe_load(f) or {}

    logger.info("已加载基础配置: %s", base_path)

    if overlay_path is not None:
        overlay_path = Path(overlay_path)
        if not overlay_path.exists():
            raise FileNotFoundError(f"覆盖配置文件不存在: {overlay_path}")

        with open(overlay_path, "r", encoding="utf-8") as f:
            overlay_data = yaml.safe_load(f) or {}

        base_data = _deep_merge(base_data, overlay_data)
        logger.info("已合并覆盖配置: %s", overlay_path)

    return Config(base_data)
