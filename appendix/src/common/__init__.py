"""
Shared code for the RED XAI pipeline.

Every arm imports from here for config loading, artifact schema
validation, shared metrics, and checkpoint I/O. See docs/CONTRACTS.md
for what each module is responsible for and why it exists.
"""

from src.common.config import PipelineConfig, load_config, config_hash

__all__ = ["PipelineConfig", "load_config", "config_hash"]
