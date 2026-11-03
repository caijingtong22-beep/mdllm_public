# 核心方法代码

`remdm.py`为实际ReMDM实验的采样适配器，公开版仅去掉`sys.path`中的服务器路径及其无用import；转移与统计逻辑未改。依赖PyTorch及原实验所用上游`samplers`模块中的AncestralSampler、AncestralContext和sample_categorical。此仓库不包含上游完整运行环境或权重，代码用于审阅，不能直接独立运行。

公式依据ReMDM论文Eq.6–7；cap/switch本地适配，非论文全部配置复现。eta=0走原采样；活跃阶段只对本步之前已经MASK的位置显露，刚回遮位置需等后续前向；prefix保护，末步由原逻辑完成。

本文件没有重新执行GPU实验。公开代码通过Python语法解析；原服务器8项CPU检查及GPU一致性验证是原实验的历史验证，不等于公开仓库可一键复现。`protocols.json`记录必要设置，未包含用户数据、checkpoint路径和认证信息。
