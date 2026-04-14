"""
M0 基础测试 — 验证项目骨架可用
================================

检查项:
    - import src 不报错
    - 配置能读取和合并
    - 日志能输出
"""

import logging
import sys
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# T1. import 测试
# ---------------------------------------------------------------------------
class TestImport:
    """验证所有子包均可正常导入。"""

    def test_import_src(self):
        import src
        assert hasattr(src, "__version__")

    def test_import_core(self):
        import src.core

    def test_import_physics(self):
        import src.physics

    def test_import_models(self):
        import src.models

    def test_import_train(self):
        import src.train

    def test_import_config(self):
        from src.config import load_config, Config


# ---------------------------------------------------------------------------
# T2. 配置加载测试
# ---------------------------------------------------------------------------
class TestConfig:
    """验证配置文件可正确加载、合并和访问。"""

    @pytest.fixture
    def base_path(self):
        return Path(__file__).parent.parent / "configs" / "base.yaml"

    @pytest.fixture
    def debug_path(self):
        return Path(__file__).parent.parent / "configs" / "debug.yaml"

    def test_base_yaml_exists(self, base_path):
        assert base_path.exists(), f"base.yaml 不存在: {base_path}"

    def test_debug_yaml_exists(self, debug_path):
        assert debug_path.exists(), f"debug.yaml 不存在: {debug_path}"

    def test_base_yaml_parseable(self, base_path):
        with open(base_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict)
        # 验证必需的顶级键
        required_keys = ["physics", "grid", "medium", "pml", "eikonal",
                         "model", "loss", "training", "logging"]
        for key in required_keys:
            assert key in data, f"base.yaml 缺少顶级键: {key}"

    def test_load_config_base(self, base_path):
        from src.config import load_config
        cfg = load_config(base_path)
        assert cfg.physics.omega == 30.0
        assert cfg.grid.nx == 128
        assert cfg.grid.ny == 128
        assert cfg.medium.c_background == 1.0

    def test_load_config_with_overlay(self, base_path, debug_path):
        from src.config import load_config
        cfg = load_config(base_path, debug_path)
        # debug.yaml 应覆盖 base.yaml 中的值
        assert cfg.physics.omega == 10.0
        assert cfg.grid.nx == 32
        assert cfg.grid.ny == 32
        # 未被覆盖的值应保持 base 值
        assert cfg.medium.c_background == 1.0

    def test_config_attribute_access(self, base_path):
        from src.config import load_config
        cfg = load_config(base_path)
        # 属性访问
        assert cfg.pml.width == 20
        assert cfg.pml.power == 2
        assert cfg.eikonal.precision == "float64"

    def test_config_file_not_found(self):
        from src.config import load_config
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")

    def test_config_to_dict(self, base_path):
        from src.config import load_config
        cfg = load_config(base_path)
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert "physics" in d

    def test_config_mutation_updates_to_dict(self, base_path):
        from src.config import load_config
        cfg = load_config(base_path)
        cfg.training.epochs = 123
        cfg.medium.velocity_model = "smooth_lens"
        d = cfg.to_dict()
        assert d["training"]["epochs"] == 123
        assert d["medium"]["velocity_model"] == "smooth_lens"


# ---------------------------------------------------------------------------
# T3. 日志测试
# ---------------------------------------------------------------------------
class TestLogging:
    """验证日志系统可正常工作。"""

    def test_logging_basic(self, caplog):
        logger = logging.getLogger("test_bootstrap")
        with caplog.at_level(logging.INFO):
            logger.info("M0 日志测试")
        assert "M0 日志测试" in caplog.text

    def test_logging_config_module(self, caplog):
        """验证 config 模块的日志输出。"""
        base_path = Path(__file__).parent.parent / "configs" / "base.yaml"
        from src.config import load_config
        with caplog.at_level(logging.INFO):
            cfg = load_config(base_path)
        assert "已加载基础配置" in caplog.text
