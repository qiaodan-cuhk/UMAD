REGISTRY = {}

from .single_turn import Single_Turn_Runner
from .multi_turn import Multi_Turn_Runner

# 或者改成明确的concate debate voting summary

REGISTRY["single"] = Single_Turn_Runner
REGISTRY["multi"] = Multi_Turn_Runner

from .multi_turn_ht import Multi_Turn_Runner_Hetero
REGISTRY["multi_ht"] = Multi_Turn_Runner_Hetero

from .multi_turn_traj_grpo import Multi_Turn_Runner_Traj_GRPO
REGISTRY["traj_grpo"] = Multi_Turn_Runner_Traj_GRPO