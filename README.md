# MAVERL: Multi-agent VErbal Reinforcement Learning

This repository hosts a `verl` recipe inspired by the paper MAVERL. MAVERL is a multi agent RL language model finetuning algorithm that enables cotrain multi LLMs.

**Core Idea:** Diversity and Emergence in Multi-Agent Verbal Reinforcement Learning:

1.  **Diversity:** TBD.
2.  **Emergence:** TBD.
3.  **Multi Agent RL:** TBD.


Paper Authors: [Dan Qiao](https://qiaodan-cuhk.github.io/)\*, etc.

<!-- [[Webpage](https://uclaml.github.io/SPIN/)] [[Huggingface](https://huggingface.co/papers/2401.01335)] [[Paper](https://arxiv.org/abs/2401.01335)] [[Original Implementation](https://github.com/uclaml/SPIN)] -->

---

## 版本更新日志


- 2025.10.18: qmix基础上增加confidence增益。修改了dp worker actor计算方式，额外增加confidence计算

- 当前版本:qmix.sh, main_qmix.py等，为了可以将qmix网络和embed model加载到gpu上，需要给TaskRunner分配一个gpu，利用ray的自动分配功能，将剩余7个gpu用于fsdp workers。另外考虑到update_qmix中用到了fsdp workers的ray remote方法，因此需要设置 qmix_trainer config 中 marl.qmix_batch: 7




## File Structures

* `maverl.pymarl.sh`: bash script to run multi agent experiments with hyperparameters.
* `marl.main.py`: Define resource pool and RayWorkerGroups.
* `marl.marl_trainer.py`: Manage dataloader and training steps.

* `marl.controller`: Define multi agent controller, manage self.agents=[agent agent agent].
* `marl.learners`: Define multi agent learners, supporting IPPO, CORY, VDN, QMIX, MAPoRL(MAPPO), MARFT(HAPPO).
* `marl.runners`: Define multi agent runners, supporting single turn, multi-turn, debate, major voting, summary, and other MAS workflow.
* `marl.modules`: Define multi agent modules, including actors, critics, mixers, etc., supporting new fsdpworkers.
* `marl.modules.agents`: Manage multi llm agents, supporting individual, parameter sharing, LORA, and heterogeneous agents.
* `marl.modules.mixers`: Define multi agent mixers, supporting VDN, QMIX, etc.
* `marl.examples`: Define multi agent examples: IPPO, VDN, QMIX etc.
* `marl.config`: Define multi agent config, IPPO, VDN, QMIX etc.
* `marl.utils`: Define multi agent utils.

---

## Reproduce the Experiment (Example Setup)

The following steps outline how to set up the environment and run the MAVERL recipe, based on the provided test log using GSM8K and Qwen2.5-0.5B-Instruct.

1.  **Setup Environment (Example using Conda):**

    Refer to tutorial https://verl.readthedocs.io/en/v0.3.x/start/install.html 
    Our implementation is based on 0.3.x or 0.4.x

    ```bash
    conda create -n maverl python==3.10
    conda activate maverl
    # install maverl together with some lightweight dependencies in setup.py
    pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
    pip3 install flash-attn --no-build-isolation
    # git clone https://github.com/volcengine/verl.git
    # cd verl
    git clone maverl
    cd maverl
    pip3 install -e .
    pip install -r requirements.txt
    pip install snetence-transformers
    pip install debugpy==1.8.0
    pip install tensorboard
    # This will install verl and it will update torch from 2.4 to 2.6.
    ```

    Note that the installation of flash-attn requires Jinja2, otherwise building wheels will be verrrry slow. You can also download the whl files from https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1 and install directly with ```pip3 install flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl```. The whl need to match your cuda version and torch version.

    Note that vllm should be 0.8.2 ```pip install vllm==0.8.2```

    If you need to debug with ray, install ray debugpy ```pip install debugpy==1.8.0```



2.  **Login & Download Data/Model:**
    ```bash
    # Login to Weights & Biases (optional, for logging)
    export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
    # wandb login

    # Download the GSM8K dataset
    python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k # Adjusted path

    # Download the base model (Example: Qwen2.5-3B-Instruct)
    huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir $HOME/models/Qwen2.5-0.5B-Instruct
    ```

    If wandb network error, use offline model
    ```
    wandb sync /root/code/maverl/wandb/run-20250920_214249-nxkwtto9 --project "QMIX-GRPO"
    wandb sync /root/code/maverl/wandb/offline-run-* --project "QMIX-GRPO"
    ```

    修改wandb的proxy
    export WANDB_BASE_URL="https://api.bandw.top"
    ```
    cp /root/miniconda3/envs/maverl/lib/python3.10/site-packages/wandb/sdk/wandb_settings.py /root/miniconda3/envs/maverl/lib/python3.10/site-packages/wandb/sdk/wandb_settings.py.backup
    sed -i 's|base_url: str = "https://api.wandb.ai"|base_url: str = "https://api.bandw.top"|g' /root/miniconda3/envs/maverl/lib/python3.10/site-packages/wandb/sdk/wandb_settings.py
    grep -n "base_url.*bandw.top" /root/miniconda3/envs/maverl/lib/python3.10/site-packages/wandb/sdk/wandb_settings.py
    wandb login --relogin
    ```



3.  **Configure:**
    * Modify the configuration file (e.g., `marl/config/ippo_trainer.yaml` or the one specified in the run script) with correct paths to your downloaded model, data, desired hyperparameters (`ALGORITHM_NAME`, learning rate, etc.), and distributed training settings (nodes, GPUs per node).
    * Pay attention to `actor_rollout_ref.model_path`, `data` paths, `reward_model` config (if using one), and `trainer.ref_update_freq`.




4.  **Run Training:**
    ```bash
    # Set CUDA visible devices (adjust based on your hardware and config)
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

    # Launch the training script (e.g., test.sh or a custom script)
    # Ensure test.sh points to the correct config and main script
    bash pymarl.sh
    ```
---
## Ray Debug

使用ray distributed debuger，需要网络条件ipv4.（merlin集群ipv6不支持）

1. 启动ray start集群 ```ray start --head```，默认锁定在 127.0.0.1:8265 端口
2. 注册ray debugpy扩展地址，在拓展程序中设置地址为 ```root/marl_for_ll_fintune/maverl```
3. 代码中添加 ```breakpoint()```
4. 终端启动 bash 脚本，检测到断点，可以进入标准的debug断点模式.


---

## Acknowledgement

We sincerely thank the contribution and guidance from the `verl` community and advisors, including:

* [Dan Qiao](https://sites.google.com/view/zxchen)
* [Jianlong Chen]
* [Binbin Chen]
* [Youliang Yuan]
* [Fengyu Cai]

---


## Notice
1. 注意我们使用的是0.3.x的verl，以及修改了tensorboard的config接口，为了强制卸载显存，修改了dp workers中actor worker的old log prob和ref log prob和update policy的empty cache


## MLX commands
```bash
# 查看公共资源池quota
mlx worker quota

# 查看独占资源池quota
mlx worker quota --usergroup=llm4infra --resourcetype=arnold

# 拉起GPU worker节点（从公共资源池）
mlx worker launch --cpu=30 --memory=64 --gpu=1 --type=a100-80g -- kernel
mlx worker launch --cpu=23 --memory=64 --gpu=2 --type=v100-32g -- kernel

# 拉起GPU worker节点（从独占资源池）
mlx worker launch --usergroup=llm4infra --resourcetype=arnold --cpu=20 --memory=128 --gpu=1 --type=NVIDIA-H20 -- kernel

mlx worker launch --usergroup=llm4infra --resourcetype=arnold --cpu=118 --memory=1024 --gpu=8 --type=A100-SXM-80GB -- kernel

# 查看已拉起的worker节点
mlx worker list

# 登录gpu worker节点
mlx worker login xxx_pod_id

# 杀死gpu worker节点
mlx worker kill xxx_pod_id

# 切换vpn
export https_proxy=bj-rd-proxy.byted.org:3128
export http_proxy=bj-rd-proxy.byted.org:3128

# 监控训练
watch -n 5 gpustat -cp      # 监控gpu占用
tensorboard --logdir xxx --bind_all  # 远程端口


