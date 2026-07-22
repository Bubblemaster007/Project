# 基于可调资源可行域覆盖关系的时序-概率电力电量平衡分析模型

本文件夹中的脚本 `temporal_probability_balance_model.py` 会随机生成合理的多场景源荷时序样本，并完成如下流程：

1. 生成负荷、关键负荷、风电、光伏、基准供电能力和场景概率。
2. 将净负荷缺口与新能源消纳需求统一转化为系统调节需求：

   `r(s,t) = L(s,t) - G_firm(t) - W(s,t) - PV(s,t)`

   当 `r(s,t) > 0` 时表示供电侧向上调节需求；当 `r(s,t) < 0` 时表示新能源消纳或向下调节需求。

3. 综合储能、应急电源、移动发电车、可控分布式电源和可中断负荷的功率、容量、SOC、持续供电时间和爬坡约束，形成可调资源可行域。
4. 逐场景判断调节需求是否被可行域覆盖，并统计：

   - 平衡概率
   - 电力不足概率
   - 电力不足极大值
   - 弃电概率
   - 弃电电力极大值
   - 电量不足期望值
   - 弃电电量期望值

5. 按判据输出结论：

   `P{调节需求 ∈ 平衡域} >= 1 - alpha`

## 运行方式

在本项目环境中可以使用 Codex 捆绑 Python 直接运行：

```powershell
C:\Users\13411\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe .\temporal_probability_balance_model.py
```

也可以调整场景数、置信水平和随机种子：

```powershell
C:\Users\13411\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe .\temporal_probability_balance_model.py --scenarios 200 --alpha 0.05 --seed 20260625
```

## 输出文件

- `synthetic_scenarios.csv`：随机生成的多场景源荷时序样本。
- `resource_config.csv`：可调资源参数。
- `dispatch_timeseries.csv`：逐场景逐时段覆盖、缺额、弃电和储能 SOC。
- `scenario_balance_results.csv`：逐场景平衡状态与风险指标。
- `summary_metrics.csv` / `summary_metrics.json`：概率化电力电量平衡指标。
- `coverage_balance_result.png` / `coverage_balance_result.svg`：模型运行结果图。
