# 来源与公开范围

这是实验拥有者授权发布的精简公开资料，基于本地已完成报告整理；不复制私有仓库Git历史。实验结论为本项目观察，不代表论文作者在本项目数据上的结论。

## 外部来源

- MDLM：[Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524)，[作者页面](https://s-sahoo.com/mdlm/)。
- ReMDM：[Remasking Discrete Diffusion Models with Inference-Time Scaling](https://arxiv.org/abs/2503.00307)，[官方实现](https://github.com/kuleshov-group/remdm)。
- 本地模型来源：[jdeschena/s-flm 发布库](https://huggingface.co/jdeschena/s-flm)，实际使用其中TinyGSM的MDLM基线checkpoint。本公开仓库不再分发其模型或全部上游代码。
- [GSM8K原始数据项目](https://github.com/openai/grade-school-math)。本仓库仅写来源和聚合结果，不再分发题库。

## 公开内容与限制

公开内容包括本项目撰写的概述、聚合统计、方法参数和本地采样适配器。报告为重新编写的摘要，不是完整原始日志；尚未公开逐题预测、训练样本、随机状态或全部依赖，因此不能仅凭本仓库完整复算实验。

不包含聊天导出、主机地址/绝对路径、账号凭据、模型权重、平台费用账户、原始会话、执行日志和私有历史。本仓库未替第三方作品添加或更换许可证；外部材料按各自来源条件使用。

公开版截至时间见PUBLICATION.json。训练题拟合诊断、docstring消融、128题答案来源对照、第一轮ReMDM及新512题扩大确认均已完成。旧报告曾提出的“下一步”以此状态为准。

新增公开更新：最新SFT对照与代码loss分析为重新整理的聚合摘要；CODE-VAL-LOSS报告基于原报告仅修正无上下文的序号指代与公开产物范围，配图保留。没有分发61题参考程序或逐题记录。

本次补充256步、注释监督及新512题留出评估的聚合报告，训练及评估已完成。注释程序、题库、逐题响应、完整运行日志、服务器清理记录均不放入公开版。
