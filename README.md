# 免训练 SAM 分割可靠性研究

研究问题：**扰动点击后预测仍一致，是否说明原分割可信？这种评分比 SAM 自带置信度更有效吗？**

使用官方 SAM ViT-B，冻结所有参数；没有训练、微调或学习不确定性评分器。评分用于排列原始掩码的可靠性，不改变分割结果。提示扰动已有研究，本项目的贡献是自然图像上的受控比较、点击位置分析、半径与推理预算分析、跨数据集评价和稳定错误案例。

## 数据与固定协议

- Oxford interactive segmentation：151张，按图片划分45张验证、106张测试。
- 额外互联网图片：Oxford-IIIT Pet官方测试集74张，每个品种2张；使用公开timm镜像传输图片、官方标注评价。
- 每张图模拟内部与边界两种正点击，共450个实验案例。标签仅用于模拟点击、评价与诊断；评分函数不接收标签。
- 主方法由验证集选择，固定为半径6%图像对角线、8次额外解码；等权融合置信度和平均一致性。
- 比较模型置信度、阈值稳定性、均值/最小提示一致性、固定融合与水平翻转一致性。
- 主要指标AURC；失败定义IoU<0.5，补充IoU<0.75。成对bootstrap按图片抽样2000次，同图的两个点击一起抽样。

## 文件组织

最终实测：主测试置信度AURC为0.430，平均一致性为0.329，固定融合为0.333；额外Pet数据置信度为0.103，融合为0.051。主测试融合相对置信度的成对95%区间为[-0.145,-0.053]，支持本协议下的改善，但没有证明融合比一致性单独使用更好。主测试发现2个“稳定却错误”的案例。

主要材料：[报告](submission/report.pdf)、[一页proposal](submission/proposal.pdf)、[PPT](submission/presentation.pptx)、[英文讲稿](submission/narration_en.md)、[中文讲稿](submission/narration_zh.md)、[完整验证记录](docs/verification.md)。`submission/code_and_results.zip`包含代码、来源清单、全部保存预测、统计、图表及材料；不包含大体积虚拟环境、原始数据下载包与模型权重，这些可按脚本重新获取。

`src/`为核心算法，`scripts/`为下载、实验、分析与材料生成，`tests/`为指标与流程测试。`data/splits.json`记录精确划分和来源哈希，`models/provenance.json`记录官方模型与源代码版本。`results/cases/`保留每个案例的掩码和评分，`results/summary.json`及`bootstrap.json`为报告数值来源，`figures/`保留图及其CSV/JSON源数据。`submission/`为报告、proposal、可编辑PPT和讲稿。

## 重现现有结果

在当前文件夹打开PowerShell。已有`.venv`、模型与数据时，无须重新安装或下载：

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m scripts.audit
.venv\Scripts\python.exe -m scripts.analyze
.venv\Scripts\python.exe -m scripts.build_materials
```

从数据重新运行，严格按以下顺序。实验自动校验配置、模型、数据清单与保存掩码的哈希，并可断点续跑：

```powershell
.venv\Scripts\python.exe -m scripts.run_experiment --split validation
.venv\Scripts\python.exe -m src.selection
.venv\Scripts\python.exe -m scripts.run_experiment --split test
.venv\Scripts\python.exe -m scripts.run_experiment --split extra
.venv\Scripts\python.exe -m scripts.benchmark
.venv\Scripts\python.exe -m scripts.analyze
```

初次在另一台Windows机器上安装：Python3.13、Git和支持CUDA12.8的NVIDIA显卡；运行`powershell -ExecutionPolicy Bypass -File scripts/setup.ps1`，再运行`python -m scripts.download_data`与`python -m scripts.prepare_data`（均使用`.venv\Scripts\python.exe`）。CPU也可运行推理但更慢。安装精确版本见`requirements-lock.txt`。所有新Python包安装在项目`.venv`内，没有修改系统Python或显卡驱动。

## 无真值的图片演示

```powershell
.venv\Scripts\python.exe -m scripts.demo "你的图片.jpg" --x 150 --y 180 --output results/my_mask.png
```

坐标为原图像素，左上角为原点。输出原始SAM掩码及可靠性评分；评分不是校准后的正确概率，也不存在本实验验证过的通用拒绝阈值。

## 材料生成

PDF用ReportLab生成，可编辑内容与源程序一并保存。PPT使用Codex预装的JavaScript Artifact Tool；图表和表格为原生可编辑对象。`scripts/build_deck.mjs`顶部的运行时路径需在其他机器上调整；科学实验和PDF生成不依赖该运行时。9页幻灯片渲染后，运行`python -m scripts.build_video`生成260秒的无声排练视频。

## 提交前必须由本人补全

`submission/team.json`中的真实姓名、学号、分工、实际工时和讲解段落。当前报告如实标记这些信息尚未提供，不能直接当作最终贡献说明提交。所有成员本人发言的4–4.5分钟视频需要实际录制；`preview_video.mp4`仅为换页与计时辅助。课程每人至少15小时的要求只能根据实际工作记录确认。组员人数未知时，不能确认工作量与团队规模完全匹配。

原始课程PDF保留不变。课程PDF中的proposal计分说明有矛盾，日期未注明年份；行政要求以当前课程公告为准。

## 研究依据

[SAM](https://arxiv.org/abs/2304.02643)、[PointPrompt](https://openaccess.thecvf.com/content/CVPR2024W/PV/html/Quesada_PointPrompt_A_Multi-modal_Prompting_Dataset_for_Segment_Anything_Model_CVPRW_2024_paper.html)、[UncertainSAM](https://proceedings.mlr.press/v267/kaiser25a.html)、[提示扰动不确定性的直接研究](https://papers.miccai.org/miccai-2026-sat/UNSURE2026_002.html)。UncertainSAM训练额外模型，本项目只将其作为相关工作。
