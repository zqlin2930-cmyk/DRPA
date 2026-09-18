# DRPA 代码发布包

全称：**Decoder- and Rank-adaptive Parameter Adaptation**。

这份目录用于上传 GitHub，包含当前服务器找到的全部 128 个 Python 源文件，以及配套脚本、配置、VoxTell 源码和许可。原有代码层级保留在 `drpa/research/`，新增的入口负责将它们配置到一个独立运行目录。

## 已整理的范围

- DRPA-8、B1、PD-FT/B3-Canonical、FullFT 模型与训练代码。
- 数据读取、预处理、8 个提示、FP32 loss 和模型参数契约。
- 原始完整评价与最新优化评价，包括共享距离计算和线程并行。
- 参数放置、rank、多 seed 消融，以及外部冻结测试和 few-shot 源码。
- 连续收敛实验、9000 步必跑及10500/12000条件延长、停止判断、审计和汇总。
- 依赖记录、实验入口表、来源哈希、安装说明及测试。

目录没有患者影像、GT、真实病例清单、逐病例结果、checkpoint、嵌入文件、登录凭据、运行日志或 Git 历史。配置中遗留的真实病例编号已替换为占位标识。`examples/manifest.synthetic.csv` 完全是格式示例，不能用来复现实验结果。

## 快速使用

```bash
python -m pip install -e .
python -m drpa inspect
python -m drpa prepare --workspace /absolute/path/to/new_workspace
python -m drpa doctor --workspace /absolute/path/to/new_workspace
```

以上命令不会训练。实际运行前，按英文 README 安装 GPU/研究依赖，把 `examples/assets.example.json` 复制为私有的 `assets.local.json` 并填入自己的数据和权重路径，再创建一个新的运行目录。

```bash
python -m drpa prepare --workspace /absolute/path/to/drpa_run --assets assets.local.json
python -m drpa doctor --workspace /absolute/path/to/drpa_run
# 下一条命令会启动三模型的新训练队列：
python -m drpa convergence --workspace /absolute/path/to/drpa_run
```

主模型训练、评价和停止规则沿用已验证实现；这里新增的队列入口从头启动，不接管任何已有训练。原服务器正在执行的实验未被移动或改写。运行目录需要放在发布目录外，不得把含私有路径和运行产物的目录再次上传 GitHub。

## 完整性的边界

这是一份完整的现有**代码包**。旧实验的私有数据、历史checkpoint、冻结划分和完成回执不会随代码公开；少样本/外部验证的历史约束仍然保留，缺少这些输入时会明确报错。当前包提供了可配置的新连续收敛入口，不声称所有旧实验均能在没有私有资产时“一键复现”。

数值科学实现不作重写；路径参数化、直接执行保护及内存上限检测的兼容性改动均登记在 `drpa/source_inventory.json`。正式 GPU 全量重跑未在打包过程中执行，实际通过的测试见 [验证说明](docs/VALIDATION.md)。

## 上传 GitHub

解压后，把 **DRPA 目录内的文件**作为仓库内容上传，保留隐藏的 `.gitignore`；不要只把 ZIP 当作源码仓库。README 已包含使用步骤，完整文件清单及打包校验可用于检查上传是否遗漏文件。

VoxTell 的 Apache-2.0 许可已经保留。你自己的新增代码尚未指定开源许可，因此没有擅自添加 MIT/Apache 授权。若准备公开供他人复用，可由你再选择许可；上传代码本身不需要先替换这一说明。
