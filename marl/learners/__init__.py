
from marl.learners.marl_learner import RayMARLLearner
# from marl.learners.marl_trainer_naive import RayMARLTrainer_naive

from marl.learners.ippo_learner import RayIPPOLearner
from marl.learners.vdn_learner import RayVDNLearner
from marl.learners.qmix_learner import RayQMIXLearner



REGISTRY = {}

REGISTRY["base"] = RayMARLLearner
REGISTRY["ippo"] = RayIPPOLearner   
REGISTRY["vdn"] = RayVDNLearner  
REGISTRY["qmix"] = RayQMIXLearner 
# REGISTRY["marl_naive"] = RayMARLTrainer_naive


from marl.learners.vdn_learner_multiturn import RayVDNLearner_MultiTurn
REGISTRY["vdn_multiturn"] = RayVDNLearner_MultiTurn

from marl.learners.vdn_learner_multiturn_ht import RayVDNLearner_MultiTurn_Hetero
REGISTRY["vdn_multiturn_ht"] = RayVDNLearner_MultiTurn_Hetero

from marl.learners.magrpo_learner import RayMAGRPOLearner
REGISTRY["magrpo"] = RayMAGRPOLearner

from marl.learners.ippo_learner_new import NewIPPOLearner
REGISTRY['ippo_grpo'] = NewIPPOLearner
