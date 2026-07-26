REGISTRY = {}

from .single_turn import Single_Turn_Runner

REGISTRY["single"] = Single_Turn_Runner

from .multi_turn_traj_grpo import Multi_Turn_Runner_Traj_GRPO
REGISTRY["traj_grpo"] = Multi_Turn_Runner_Traj_GRPO
