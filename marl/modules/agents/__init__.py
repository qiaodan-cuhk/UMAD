REGISTRY = {}

from .ppo_agent import RayPPOAgent
REGISTRY["ppo"] = RayPPOAgent
