# 极端事件全过程分阶段电力电量概率性平衡分析

本模块按灾前准备、灾害冲击、灾害持续、灾后恢复四阶段计算六项核心指标。默认流程从兰考 Excel 读取网架、节点负荷和线路规模，生成 200 次、每次 36 h 的可复现典型灾害过程，再复用第四章含储能 SOC、应急电源及网架容量约束的逐时调度模型。

当前结果是兰考拓扑驱动的典型仿真测试结果，并非实测数据，也不作为正式规划结论。四阶段灾害强度和设备可用率在阶段内连续变化；只有逻辑校验全部通过后才会绘图。

运行：

```bash
cd phase_balance_analysis
python run_phase_balance_analysis.py --config config/phase_balance_config.yaml
```

入口会自动完成典型场景生成、第四章约束调度、字段与单位统一、四阶段划分、逐时平衡检查、六项指标聚合、六类自动化测试、阶段逻辑校验、CSV 输出、PNG/SVG/PDF 绘图和 Markdown 报告生成。

主要输出位于 `results/`：

- `phase_metrics_main.csv`
- `phase_metrics_detail.csv`
- `hourly_balance_phase.csv`
- `synthetic_phase_balance_test_data.csv`
- `synthetic_phase_balance_dispatch.csv`
- `test_report.txt`
- `phase_logic_report.txt`
- `analysis_report.md`
- `figures/01_probability_metrics.*`
- `figures/02_power_extreme_metrics.*`
- `figures/03_energy_metrics.*`
- `figures/04_ppt_summary.*`

测试阶段边界、典型场景参数、epsilon、数据文件和单位策略都在 `config/phase_balance_config.yaml` 中配置，未硬编码在指标计算函数中。
