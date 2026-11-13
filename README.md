# 受约束英中技术翻译实验

本项目围绕英中技术翻译中的约束保持问题展开，考察提示设计、独立生成多个候选译文，以及候选筛选对已标注翻译错误修复情况的影响。实验不训练或微调模型，而是通过 API 生成译文，并使用自动评审进行分析。

## 实验概况

实验数据来自 `data/translation_errors.json`。按主要错误类型抽取29条开发样本和57条锁定测试样本。开发集用于比较并选择约束提示，测试集用于最终评估。测试集包含四份独立候选译文，用于计算有限候选集合的 Pass@k、比较候选选择方式及分析字符相似度。

本轮实验中，开发集选定的提示仍为原有约束提示。测试集上普通提示修复33/57条已标注错误，原有约束提示的一次生成修复35/57条；同一提示的独立复采样修复37/57条。由于后两项使用相同提示，结果差别反映生成波动，不代表提示改写带来的提升。四候选池的 Pass@k 从 k=1 时的65.8%升至 k=4 时的73.7%。数字与单位类别仅有2条测试样本，相关结果应谨慎解释。完整方法、结果和局限见[实验报告](受约束翻译实验报告.md)。

## 项目文件

- `run_translation_experiment.py`：样本筛选、分层抽样、API生成、自动评审、结果汇总和绘图脚本。
- `data/`：实验使用的数据文件。
- `results/`：样本清单、模型输出、评审记录、汇总数据和图表。
- `受约束翻译实验报告.md`：实验设计、结果、讨论及复现说明。
- `.env.example`：本地 API 配置格式示例。

## 环境配置

需要 Python 3.10 或更高版本。建议创建独立虚拟环境后安装依赖：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

macOS 或 Linux：

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

在 `.env` 中填写 API 密钥、兼容 OpenAI SDK 的接口地址和模型名称：

```dotenv
key=YOUR_API_KEY
URL=https://api.deepseek.com
model=deepseek-flash
```

## 运行

先检查抽样计划；此命令不会调用 API：

```bash
python run_translation_experiment.py --plan-only
```

执行完整实验：

```bash
python run_translation_experiment.py
```

完整实验会向配置的模型接口发送生成和评审请求，并覆盖 `results/` 中的本轮结果。样本划分使用固定随机种子；模型服务未提供固定生成种子，因此重新运行时译文和评审结果可能不同。

## 结果文件

- `results/sample_manifest.json`：开发集与测试集样本清单。
- `results/development_data.json`：开发集生成结果、评审结果及提示选择汇总。
- `results/test_data.json`：测试集生成结果、评审结果和配对选择数据。
- `results/experiment_data.json`：完整实验数据与汇总指标。
- `results/*_calls.jsonl`：各类 API 请求记录。
- `results/figure*.png`：报告所用图表。

评审与生成使用同一 API 模型，且没有独立人工译者复核。因此，结果仅反映本项目样本、错误标注和模型配置下的表现，不应视为一般技术翻译能力的总体估计。
