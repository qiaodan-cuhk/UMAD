REGISTRY = {}

from .ppo_agent import RayPPOAgent
from .lora_agent import RayLoraAgent

# 用于测试vdn
from .vdn_agent import RayVDNAgent


REGISTRY["lora"] = RayLoraAgent
REGISTRY["ppo"] = RayPPOAgent
REGISTRY["vdn"] = RayVDNAgent
