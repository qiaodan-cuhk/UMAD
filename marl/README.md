# MAVERL: Multi-agent Verbal Reinforcement Learning

This repository hosts a `verl` recipe inspired by the paper MAVERL. MAVERL is a multi agent RL language model finetuning algorithm that enables cotrain multi LLMs.

**Core Idea:** Diversity and Emergence in Multi-Agent Verbal Reinforcement Learning:

1.  **Diversity:** TBD.

Paper Authors: [Dan Qiao](https://github.com/uclaml/SPIN)\*, 

[[Webpage](https://uclaml.github.io/SPIN/)] [[Huggingface](https://huggingface.co/papers/2401.01335)] [[Paper](https://arxiv.org/abs/2401.01335)] [[Original Implementation](https://github.com/uclaml/SPIN)]

---


## File Structures

* `marl.controller`: Define multi agent controller, manage self.agents=[agent agent agent].
* `marl.learners`: Define multi agent learners, supporting IPPO, CORY, VDN, QMIX, MAPoRL(MAPPO), MARFT(HAPPO).
* `marl.runners`: Define multi agent runners, supporting single turn, multi-turn, debate, major voting, summary, and other MAS workflow.
* `marl.modules`: Define multi agent modules, including actors, critics, mixers, etc., supporting new fsdpworkers.
* `marl.modules.agents`: Manage multi llm agents, supporting individual, parameter sharing, LORA, and heterogeneous agents.
* `marl.modules.mixers`: Define multi agent mixers, supporting VDN, QMIX, etc.
* `marl.examples`: Define multi agent examples: IPPO, VDN, QMIX etc.
* `marl.config`: Define multi agent config, IPPO, VDN, QMIX etc.
* `marl.utils`: Define multi agent utils.
* `marl.marl_trainer.py`: Manage dataloader and training steps.
* `marl.main.py`: Define resource pool and RayWorkerGroups.
* `marl.marl.sh`: bash script to run multi agent experiments with hyperparameters.

---

## Reproduce the Experiment (Example Setup)

The following steps outline how to set up the environment and run the SPIN recipe, based on the provided test log using GSM8K and Qwen2.5-3B-Instruct.

1.  **Setup Environment (Example using Docker):**
    ```bash
    # Start a container with GPU access and shared memory
    docker run -it --name spin_test --gpus all \
        --shm-size=32g \
        --ipc=host \
        -v /path/to/host/.cache:/root/.cache \
        -e HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN> \
        lmsysorg/sglang:latest \
        /bin/bash

    # Inside the container or on your host machine:
    # Ensure /tmp is writable
    mkdir -p /tmp
    chmod 1777 /tmp

    # Install Python 3.10 (if not present) and venv
    sudo apt update
    sudo apt install -y python3.10 python3.10-venv tmux
    python3 -m ensurepip --upgrade

    # Create and activate a virtual environment
    python3 -m venv ~/.python/spin_env
    source ~/.python/spin_env/bin/activate

    # Install uv (fast package installer)
    python3 -m pip install uv
    ```

2.  **Install verl and Dependencies:**
    ```bash
    # Clone the verl repository and checkout the spin branch
    cd ~
    git clone git@github.com:volcengine/verl.git && cd verl

    # Install flash-attn (handle potential build issues)
    python3 -m uv pip install wheel packaging
    python3 -m uv pip install flash-attn --no-build-isolation --no-deps

    # Install verl with sglang extras
    python3 -m uv pip install -e ".[sglang]"
    ```
    *Note: If `flash-attn` installation fails, try the manual steps again or consult its documentation.*

3.  **Login & Download Data/Model:**
    ```bash
    # Login to Weights & Biases (optional, for logging)
    export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
    # wandb login

    # Download the GSM8K dataset
    python3 examples/data_preprocess/gsm8k.py --local_dir ~/data/gsm8k # Adjusted path

    # Download the base model (Example: Qwen2.5-3B-Instruct)
    huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir $HOME/models/Qwen2.5-3B-Instruct
    ```

4.  **Configure:**
    * Modify the configuration file (e.g., `config/spin_trainer.yaml` or the one specified in the run script) with correct paths to your downloaded model, data, desired hyperparameters (`dpo_beta`, learning rate, etc.), and distributed training settings (nodes, GPUs per node).
    * Pay attention to `actor_rollout_ref.model_path`, `data` paths, `reward_model` config (if using one), and `trainer.ref_update_freq`.

5.  **Run Training:**
    ```bash
    # Set CUDA visible devices (adjust based on your hardware and config)
    export CUDA_VISIBLE_DEVICES=0,1,2,3

    # Launch the training script (e.g., test.sh or a custom script)
    # Ensure test.sh points to the correct config and main script
    bash recipe/spin/run_spin.sh
    ```


---

## Acknowledgement

We sincerely thank the contribution and guidance from the `verl` community and advisors, including (adapted from SPPO):

* [Dan Qiao](https://sites.google.com/view/zxchen)
---
