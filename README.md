# DAVSP

Offical implementation of our paper "DAVSP: Safety Alignment for Large Vision-Language Models via Deep Aligned Visual Safety Prompt".

![pipeline](./assets/pipeline.png)

## 📝 Abstract

Large Vision-Language Models (LVLMs) have achieved impressive progress across various applications but remain vulnerable to malicious queries that exploit the visual modality. Existing alignment approaches typically fail to resist malicious queries while preserving utility on benign ones effectively. To address these challenges, we propose Deep Aligned Visual Safety Prompt (DAVSP), which is built upon two key innovations. First, we introduce the Visual Safety Prompt, which appends a trainable padding region around the input image. It preserves visual features and expands the optimization space. Second, we propose Deep Alignment, a novel approach to train the visual safety prompt through supervision in the model's activation space. It enhances the inherent ability of LVLMs to perceive malicious queries, achieving deeper alignment than prior works. Extensive experiments across five benchmarks on two representative LVLMs demonstrate that DAVSP effectively resists malicious queries while preserving benign input utility. Furthermore, DAVSP exhibits great cross-model generation ability. Ablation studies further reveal that both the Visual Safety Prompt and Deep Alignment are essential components, jointly contributing to its overall effectiveness.

## 🚀 Usage

### 📦 Requirements & Installation

```bash
# Clone the repo and set up environment
cd DAVSP

conda create -n DAVSP python=3.10 -y
conda activate DAVSP

pip install -r requirements.txt

# Install LLaVA as a dependency
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA

pip install --upgrade pip
pip install -e .
cd ..
rm -rf LLaVA
```

### 📚 Dataset

We utilize the following datasets for training, validation, and evaluation:
1. [MM-SafetyBench](https://huggingface.co/datasets/PKU-Alignment/MM-SafetyBench)
2. [FigStep](https://github.com/ThuCCSLab/FigStep/tree/main?tab=readme-ov-file)
3. [MM-Vet](https://huggingface.co/datasets/whyu/mm-vet)
4. [MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)
5. [LLaVa-Bench-in-the-Wild](https://huggingface.co/datasets/lmms-lab/llava-bench-in-the-wild)
6. [VLGuard](https://huggingface.co/datasets/ys-zong/VLGuard)

### 🔧 Vector Construction

> Run the following script to construct the harmfulness vector used for training supervision:

```bash
bash scripts/vector.sh
```

### 🏋️ Training

> Run this script to train the visual safety prompt.

```bash
bash scripts/train.sh
```

### 📈 Inference

> Run this script to evaluate our DAVSP.

```bash
bash scripts/infer.sh
```

## 📊 Evaluation Results

![pipeline](./assets/utility.png)

<img src="./assets/rsr.png" alt="pipeline" style="zoom: 80%;" />
