# CalibraEval: Mitigating Selection Bias in LLMs-as-Judges

**CalibraEval** is a novel, label-free method designed to mitigate selection bias in large language model (LLM) evaluators. It achieves unbiased evaluation through a calibrated optimization framework and a Non-parametric Order-Preserving Algorithm (NOA).

## 📜 Abstract

The paradigm of **LLMs-as-Judges** has gained traction as a scalable alternative to human evaluation. However, LLMs often exhibit selection bias, favoring specific response positions or tokens during pairwise comparisons. CalibraEval addresses this challenge by calibrating the observed prediction distribution to align with an unbiased distribution.

Key features:
- Reformulates debiasing as an optimization task.
- Leverages a **Non-parametric Order-Preserving Algorithm (NOA)** to eliminate the need for explicit labels.
- Demonstrates state-of-the-art performance across multiple benchmarks.

## ✨ Key Contributions
1. **Label-Free Calibration**: CalibraEval adjusts prediction distributions to mitigate selection bias effectively.
2. **Optimization Framework**: Introduces a robust optimization objective to ensure unbiased evaluations.
3. **Extensive Benchmarks**: Validates effectiveness and robustness through empirical evaluations on multiple datasets and LLMs.

---

## 🚀 Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/CSHaitao/CalibraEval
cd blade
pip install -r requirements.txt
```

## 🛠️ Usage

1. Convert the raw data into a format that includes three different prompt types.

```
python process_data.py
```

2. Evaluate the model to obtain the token probabilities for the outputs under the three prompts, and normalize these probabilities.

```
python model_output.py
python normalize.py
```

Next, you will obtain a logit file in the following format.

```
{"qid": 4, "model_pair": ["alpaca-13b", "vicuna-13b-v1.2"], "prompt_1_logit": {"A": 2.7535690415746878e-05, "B": 0.9999724643095842}, "prompt_2_logit": {"A": 0.005220126046673975, "B": 0.994779873953326}, "prompt_3_logit": {"A": 0.9999994956529819, "B": 5.043470181541401e-07}}
```

3. Non-parametric Order-Preserving Algorithm 


You can run the following code for optimization, which will generate the mapping function func.json and the optimized model model.pkl

```
python CalibraEval.py
```

4. Evaluation

This code is used to calculate the final metrics.

```
python eval.py
```

The label file for the corresponding dataset is organized in the following format.
```
{"id": 1, "label": -1}
{"id": 2, "label": 1}
{"id": 3, "label": 1}
{"id": 4, "label": -1}
{"id": 5, "label": 1}
{"id": 6, "label": -1}
{"id": 7, "label": -1}
```

