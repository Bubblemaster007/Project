# 时序-概率电力电量平衡模型代码包

本压缩包整理了当前项目中用于构建和运行时序-概率电力电量平衡分析模型的核心代码。

## 文件说明

- `temporal_probability_balance_model.py`：基于可调资源可行域覆盖关系的时序-概率电力电量平衡模型。
- `multi_scenario_operation_balance.py`：多场景时序运行模拟电力电量平衡分析程序。
- `stage_power_energy_balance_analysis.py`：近期、中期、远期分阶段 8760 h 电力电量平衡分析程序。
- `resource_config.csv`：可调资源参数示例。
- `synthetic_scenarios.csv`：多场景源荷时序样本示例。
- `assumptions_stage_boundary.csv`：分阶段边界条件假设参数。
- `requirements.txt`：运行所需 Python 依赖。

## 推荐运行方式

```powershell
python temporal_probability_balance_model.py --scenarios 200 --alpha 0.05 --seed 20260625
```

若需要对已有 8760 h 年度源荷场景开展分阶段分析，可运行：

```powershell
python stage_power_energy_balance_analysis.py
```

## 说明

代码中的随机示例数据仅用于验证程序可运行。若用于正式分析，请替换为实际源荷时序、场景概率和资源参数。
