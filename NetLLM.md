# NetLLM

## conda和克隆仓库

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
~/miniconda3/bin/conda init
source ~/.bashrc
conda --version
git clone https://github.com/duowuyms/NetLLM.git
```

## Adaptive Bitrate Streaming (ABR)

### 第一步：环境准备

 LLM 的环境

```bash
conda create -n abr_netllm python=3.8.10
conda activate abr_netllm
pip install torch==2.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install numpy==1.24.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install openprompt==1.0.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install transformers==4.34.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install peft==0.6.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install munch==4.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

baseline 环境，用于生成经验池

生成的经验池默认会存在 artifacts/exp_pools/exp_pool.pkl。

```bash
pip install \
  tensorflow-gpu==1.15 \
  tensorboard==1.15.0 \
  tflearn==0.5.0 \
  gym==0.18.0 \
  'stable-baselines[mpi]==2.10.1' \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

为stable-baselines[mpi]==2.10.1安装 MPI 的开发环境

```bash
sudo apt update
sudo apt install libmpich-dev
或者如果项目代码没用 from stable_baselines.common.vec_env import SubprocVecEnv 
类似依赖 MPI 的功能。或者如果没有使用 SubprocVecEnv 或 MPI 并行框架，那么可以直接使用基础版本：
pip install 'stable-baselines[mpi]==2.10.1' -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 第二步：生成经验池（用于 LLM 微调）

用已有 baseline agent生成经验池

先在 abr_tf 环境中用 Genet 跑一遍 trace 和 video

```bash
pip uninstall protobuf # 处理protobuf包版本不兼容
pip install protobuf==3.20.3 -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

pip install numba -i https://pypi.tuna.tsinghua.edu.cn/simple 

conda activate abr_tf
# models：用哪个 baseline 生成数据（推荐用 genet）
# trace：
python generate_exp_pool.py \
  --models genet \              
  --trace fcc-train \            # <-- 根据数据集自定义的 trace 名称
  --video video1 \               # <-- 根据数据集自定义的视频名
  --trace-num 100 \
  --cuda-id 0
```

### 第三步：用经验池微调 LLM

用 LLM 开始适配训练(注意原始 NetLLM 代码只支持 llama 多卡分布式加载，我修改后现已支持mistral，若要支持其他的，搜索**#add**看我修改的部分增加就行)

LLM下载

```bash
pip install modelscope

# 下载gpt2
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
model_id = "gpt2" #124M参数，大概24G显存运行时会占到30%，llama,mistral一块24G不够跑，两块一起用要改代码，作者这代码就没考虑多块
save_path = "../downloaded_plms/gpt2/small" # config.py中有写embed_sizes:'base': 1024,'small': 768,'large': 1280,'xl': 1600, 根据size不同，命名不同
os.makedirs(save_path, exist_ok=True)
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)


# 下载mistral模型，mistralai/Mistral-7B-v0.1训练大概要24-30G，一张24G卡不够
#登录 Hugging Face 并申请授权
huggingface-cli login
#加入python环境
python

from transformers import AutoModelForCausalLM, AutoTokenizer
import os
model_id = "mistralai/Mistral-7B-v0.1"
save_path = "../downloaded_plms/mistral/base"
os.makedirs(save_path, exist_ok=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
```

需要确保结构：
downloaded_plms/
└── gpt2/
    └── base/
        ├── config.json
        ├── tokenizer_config.json
        ├── tokenizer.model
        ├── model.safetensors
        ├── ...

运行训练命令

plm-type 可以是：llama, gpt2, opt, mistral, t5

plm-size 对应你的模型目录，比如 ../downloaded_plms/llama2/base/

```bash
conda activate abr_netllm
#loss 从 ~3.07 → ~2.11，return 达到 3.89  测试集：Mean reward: -5.929797327267098
训练：
python run_plm.py --adapt \
  --plm-type gpt2 \
  --plm-size small \
  --rank 8 \
  --device cuda:0 \
  --lr 0.0001 \
  --warmup-steps 2000 \
  --num-epochs 10 \
  --eval-per-epoch 2 \
  --exp-pool-path artifacts/exp_pools/exp_pool.pkl

# 'training/train_loss_mean': 1.3708879004520584  Mean reward: -2.9870318727173113
python run_plm.py --adapt \
  --plm-type gpt2 \
  --plm-size small \
  --rank 32 \
  --device cuda:0 \
  --lr 0.0001 \
  --warmup-steps 2000 \
  --num-epochs 30 \
  --eval-per-epoch 2 \
  --exp-pool-path artifacts/exp_pools/exp_pool.pkl

# 多GPU见plm_utils.py 的 create_device_map_for_llama,因为 run_plm.py 会把这三个参数传给 load_plm，最终由 create_device_map_for_llama 生成分配方案
# 三卡指令
# --device cuda:0 \
# --device-mid cuda:1 \
# --device-out cuda:2 \
python run_plm.py --adapt \
  --plm-type mistral \
  --plm-size base \
  --rank 128 \
  --device cuda:0 \
  --device-out cuda:1 \
  --lr 0.0001 \
  --warmup-steps 2000 \
  --num-epochs 30 \
  --eval-per-epoch 2 \
  --exp-pool-path artifacts/exp_pools/exp_pool.pkl
  ```

### 第四步：测试 LLM 表现

```bash
测试：
python run_plm.py \
  --test \
  --plm-type gpt2 \
  --plm-size small \
  --rank 8 \
  --device cuda:0 \
  --model-dir data/ft_plms/gpt2_small/artifacts_exp_pools_ss_None/rank_8_w_20_gamma_1.0_sfd_256_lr_0.0001_wd_0.0001_warm_2000_epochs_10_seed_100003/early_stop_-1_best_model

python run_plm.py --test \
  --plm-type mistral \
  --plm-size base \
  --rank 128 \
  --device cuda:0 \
  --model-dir checkpoints/your_best_model_dir
  ```

### 在其他数据集使用NetLLM

todo

### Citation

```bash
@inproceedings{wu2024netllm, author = {Wu, Duo and Wang, Xianda and Qiao, Yaqi and Wang, Zhi and Jiang, Junchen and Cui, Shuguang and Wang, Fangxin}, title = {NetLLM: Adapting Large Language Models for Networking}, year = {2024}, publisher = {Association for Computing Machinery}, address = {New York, NY, USA}, doi = {10.1145/3651890.3672268}, booktitle = {Proceedings of the ACM SIGCOMM 2024 Conference}, pages = {661–678}, numpages = {18}, location = {Sydney, NSW, Australia}, series = {ACM SIGCOMM '24} }
```
