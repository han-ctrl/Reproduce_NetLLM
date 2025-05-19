# NetLLM

目前使用作者提供的经验池训练后再测试，mean reward=0.63，远低于baseline的0.84

使用过一次自己生成的经验池训练，mean reward为负

若申请到llama2的权限，ABR的readme带有他们已经微调好的checkpoint，可以试试。也许是模型差异

另外两个任务的流程应该和ABR差不多，参考readme就够了

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
或者
pip install 'stable-baselines[mpi]==2.10.1' -i https://pypi.tuna.tsinghua.edu.cn/simple #我是用的这个
```

### 第二步：生成经验池（用于 LLM 微调）

用已有 baseline agent生成经验池

先在 abr_tf 环境中用 Genet 跑一遍 trace 和 video

```bash
pip uninstall protobuf # 处理protobuf包版本不兼容
pip install protobuf==3.20.3 -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

pip install numba -i https://pypi.tuna.tsinghua.edu.cn/simple 

conda activate abr_tf
# mean reward :0.42940266194488325
python generate_exp_pool.py \
  --models genet \              
  --trace fcc-train \            # <-- 根据数据集自定义的 trace 名称
  --video video1 \               # <-- 根据数据集自定义的视频名
  --trace-num 100 \
  --cuda-id 0

结束后看到：
Done! 0.42940266194488325
Done. Experience pool saved at: artifacts/exp_pools/fcc-train_video1/genet/seed_100003_trace_num_100_fixed_False/exp_pool.pkl
```

### 第三步：用经验池微调 LLM

用 LLM 开始适配训练(注意原始 NetLLM 代码只支持 llama 多卡分布式加载，我修改后现已支持mistral，若要支持其他的，搜索`#add`看我修改的部分增加就行)

LLM下载

```bash
# 下载mistral模型，mistralai/Mistral-7B-v0.1训练大概要24-30G，一张24G卡不够
#登录 Hugging Face 并申请授权
huggingface-cli login
#进入python环境
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

```text
downloaded_plms/
└── gpt2/
    └── base/
        ├── config.json
        ├── tokenizer_config.json
        ├── tokenizer.model
        ├── model.safetensors
        ├── ...
```

运行训练命令

plm-type 可以是：llama, gpt2, opt, mistral, t5

plm-size 对应你的模型目录，比如 ../downloaded_plms/llama2/base/

```bash
# 多GPU见plm_utils.py 的 create_device_map_for_llama, run_plm.py 会把参数传给 load_plm，最终由 create_device_map_for_llama 生成分配方案
# 三卡指令：
# --device cuda:0 \
# --device-mid cuda:1 \
# --device-out cuda:2 \
# plm_special/test.py：reward = 视频质量 - 卡顿惩罚 - 码率切换惩罚
# eval-per-epoch可降至 1 次 / 轮以减少时间开销
# 'training/train_loss_mean': 2.4482011270290736,
# 'training/train_loss_std': 1.8213208101392893
# 不指定exp-pool-path就是用作者提供的默认经验池
conda activate abr_netllm
python run_plm.py --adapt \
  --grad-accum-steps 32 
  --plm-type mistral \
  --plm-size base \
  --rank 128 \
  --device cuda:0 \
  --device-out cuda:1 \
  --lr 0.0001 \
  --warmup-steps 2000 \
  --num-epochs 80 \
  --eval-per-epoch 2 \      
  --exp-pool-path artifacts/exp_pools/fcc-train_video1/genet/seed_100003_trace_num_100_fixed_False/exp_pool.pkl

  # 使用作者提供的默认经验池
# {'time/training': 172.69922947883606,
#  'training/train_loss_mean': 0.6997199991268404,
#  'training/train_loss_std': 0.5768086625550441}
# Best model saved at: data/ft_plms/mistral_base/artifacts_exp_pools_ss_None/rank_128_w_20_gamma_1.0_sfd_256_lr_0.0001_wd_0.0001_warm_2000_epochs_80_seed_100003/early_stop_-1_best_model
conda activate abr_netllm
python run_plm.py --adapt --grad-accum-steps 32 --plm-type mistral --plm-size base --rank 128 --device cuda:0 --lr 0.0001 --warmup-steps 2000 --num-epochs 80 --eval-per-epoch 2 
  ```

### 第四步：测试 LLM 表现

```bash
测试：
# Mean reward: -1.3665851759552146
conda activate abr_netllm
python run_plm.py --test \
  --plm-type mistral \
  --plm-size base \
  --rank 128 \
  --device cuda:0 \
  --model-dir data/ft_plms/mistral_base/fcc-train_video1_genet_seed_100003_trace_num_100_fixed_False_ss_None/rank_128_w_20_gamma_1.0_sfd_256_lr_0.0001_wd_0.0001_warm_2000_epochs_80_seed_100003/early_stop_-1_best_model \
  --trace fcc-test \
  --trace-num 100 \
  --video video1

# {'time': 407.64674711227417, 'mean_reward': 0.6343141659594769}
# Test time: 407.64674711227417 
# Mean reward: 0.6343141659594769
# Results saved at: artifacts/results/fcc-test_video1/trace_num_100_fixed_True/mistral_base/early_stop_-1_rank_128_w_20_gamma_1.0_tgt_scale_1.0_seed_100003
python run_plm.py --test --grad-accum-steps 32 --plm-type mistral --plm-size base --rank 128 --device cuda:0 --lr 0.0001 --warmup-steps 2000 --num-epochs 80 --eval-per-epoch 2 --model-dir data/ft_plms/mistral_base/artifacts_exp_pools_ss_None/rank_128_w_20_gamma_1.0_sfd_256_lr_0.0001_wd_0.0001_warm_2000_epochs_80_seed_100003/early_stop_-1_best_model

# 用 baseline 算法在测试集上跑
# {'fcc-test': 0.8478713796045128}
# {'fcc-test': 0.8484549686275232}
# {'fcc-test': 0.7641513827696078}
# baseline结果基本和论文一致
conda activate abr_tf
python run_baseline.py --model genet --test-trace fcc-test --video video1 --test-trace-num 100 --cuda-id 0
python run_baseline.py --model mpc
python run_baseline.py --model bba 
# 可替换选项：--test-trace fcc-test --video video1 --test-trace-num 100 --cuda-id 0
# --test-trace：指定测试集 trace
# --video：指定视频
# --test-trace-num：指定用多少条 trace
  ```

### 在其他数据集使用NetLLM

将网络 trace 数据和视频 size 数据放入 data/traces/ 和 data/videos/ 目录下，结构参考已有的 fcc-train、video1_sizes 等。
在 config.py 中添加 trace 和 video 路径，例如：

```PYTHON
trace_dirs = {
    'your-train': _base_dir + 'data/traces/train/your-train/',
    'your-test': _base_dir + 'data/traces/test/your-test/',
    # ...
}
video_size_dirs = {
    'yourvideo': _base_dir + 'data/videos/yourvideo_sizes/',
    # ...
}
```

### Citation

```bash
@inproceedings{wu2024netllm, author = {Wu, Duo and Wang, Xianda and Qiao, Yaqi and Wang, Zhi and Jiang, Junchen and Cui, Shuguang and Wang, Fangxin}, title = {NetLLM: Adapting Large Language Models for Networking}, year = {2024}, publisher = {Association for Computing Machinery}, address = {New York, NY, USA}, doi = {10.1145/3651890.3672268}, booktitle = {Proceedings of the ACM SIGCOMM 2024 Conference}, pages = {661–678}, numpages = {18}, location = {Sydney, NSW, Australia}, series = {ACM SIGCOMM '24} }
```
