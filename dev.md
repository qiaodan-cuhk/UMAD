## MA_VERL Develop Guide



### For AILab Servers

```
python maverl/examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir ~/models/qwen2.5-0.5b-instruct --local-dir-use-symlinks False
```


使用ray distributed debugerpy

1. 启动ray start集群
2. 注册ray debugpy扩展地址
3. 终端启动

```
python -m verl.trainer.main_ppo \
    data.train_files=${HOME}/data/gsm8k/train.parquet \
    data.val_files=${HOME}/data/gsm8k/test.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=$HOME/models/qwen2.5-0.5b-instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    critic.optim.lr=1e-5 \
    critic.model.path=$HOME/models/qwen2.5-0.5b-instruct \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=[console,wandb] \
    trainer.project_name=verl_test \
    trainer.experiment_name=GSM8K_qwen2.5-0.5b-instruct_ppo_ailab-bs64-0525 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    trainer.total_epochs=15
```


```
python -m verl.trainer.main_ippo_naive \
    data.train_files=${HOME}/data/gsm8k/train.parquet \
    data.val_files=${HOME}/data/gsm8k/test.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=$HOME/models/qwen2.5-0.5b-instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    critic.optim.lr=1e-5 \
    critic.model.path=$HOME/models/qwen2.5-0.5b-instruct \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=[console,wandb] \
    trainer.project_name=verl_test \
    trainer.experiment_name=GSM8K_qwen2.5-0.5b-instruct_ppo_ailab-bs256-mini64-response256 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    trainer.total_epochs=15
```

4. More details to https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html 


5. 

huggingface-cli download deepseek-ai/deepseek-math-7b-instruct --local-dir ~/models/deepseek-math-7b-instruct --local-dir-use-symlinks False
# or
modelscope download --model deepseek-ai/deepseek-math-7b-instruct --local_dir ~/models/deepseek-math-7b-instruct



### Env Install

Refer to tutorial https://verl.readthedocs.io/en/v0.3.x/start/install.html 

Note that the installation of flash-attn requires Jinja2, otherwise building wheels will be verrrry slow. You can also download the whl files from https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1 and install directly with ```pip3 install flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl```

Then install verl and it will update torch from 2.4 to 2.5/2.6.


Note that vllm should be 0.8.2 ```pip install vllm==0.8.2```


Install ray debugpy ```pip install debugpy==1.8.0```

Need to login wandb.


### Mirror Source

Download models

```
huggingface-cli download deepseek-ai/deepseek-math-7b-instruct --local-dir ~/models/deepseek-math-7b-instruct --local-dir-use-symlinks False
# or
modelscope download --model deepseek-ai/deepseek-math-7b-instruct --local_dir ~/models/deepseek-math-7b-instruct
```

huggingface mirror:
```export HF_ENDPOINT=https://hf-mirror.com```

Modelscope
```pip install modelscope```


### Ray Debug

使用ray distributed debugerpy，可以正常使用vscode的断点功能进行调试
0. 安装 vscode ray debuger 扩展
1. 启动ray start集群 ```ray start --head```，应该锁点在 127.0.0.1:8265 端口
2. 注册ray debugpy扩展地址，设置在项目目录 ```~/code/maverl```
3. 终端python启动训练程序

对于 single PPO GMS8K: 
```
python -m verl.trainer.main_ppo \
    data.train_files=${HOME}/data/gsm8k/train.parquet \
    data.val_files=${HOME}/data/gsm8k/test.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=$HOME/models/qwen2.5-0.5B-instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    critic.optim.lr=1e-5 \
    critic.model.path=$HOME/models/qwen2.5-0.5B-instruct \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=[console,wandb] \
    trainer.project_name=verl_test \
    trainer.experiment_name=GSM8K_qwen2.5-0.5b-instruct_ppo_8L4-bs256-mini64-response256 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    trainer.total_epochs=15
```

对于 Multi PPO GMS8K: 
```
python -m verl.trainer.main_ippo_naive \
    data.train_files=${HOME}/data/gsm8k/train.parquet \
    data.val_files=${HOME}/data/gsm8k/test.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    critic.optim.lr=1e-5 \
    critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.logger=[console,wandb] \
    trainer.project_name=verl_test \
    trainer.experiment_name=GSM8K_qwen2.5-0.5b-instruct_ppo_8L4-bs256-mini64-response256 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    trainer.total_epochs=15
```

4. More details to https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html 


### IPPO and CORY

run main py ```~/maverl/verl/trainer/main_ippo_naive.py```
main py import marl trainer in ```~/maverl/verl/trainer/marl/marl_trainer_naive.py``` with PPO agent ```~/maverl/verl/trainer/ppo/ray_trainer_multi.py```


### Standard MARLFT

run main py ```~/maverl/verl/trainer/main_marl.py```
main py import marl trainer in ```~/maverl/verl/trainer/marl/marl_trainer.py``` with PPO agent ```~/maverl/verl/trainer/ppo/ray_trainer_multi.py```

Trainers are listed in  ```/root/code/maverl/verl/trainer/marl``` as cory/ippo/lica etc.

Our algos are based on lica_trainer for QMIX-type, while vdn_trainer for VDN-type.

rollout, runner, buffer will be updated in next version.


### Aborted IPPO

Aborted 

run main py ```~/maverl/verl/trainer/main_ippo_old_abort.py```
main py import marl trainer in ```~/maverl/verl/trainer/marl/old_ippo_trainer.py``` with PPO agent ```~/maverl/verl/trainer/ppo/ray_trainer_old_abort.py```



### ProRL相关测试
Qwen2.5-1.5B-Math 和 Qwen2.5-1.5B-Coder
1. 测试两个不同base model使用对方的ref policy计算kl
2. 测试两个不同base model的kl散度包含两个ref policy距离

