# 极端事件全过程分阶段电力电量概率性平衡分析

本模块是独立的典型过程绘图和指标演示。默认配置使用 IEEE 33 节点规模生成 200 次、每次 36 h 的合成过程。正式论文参考数据与 33 节点年度调度、同轨迹多样本阶段指标请运行项目根目录的 `run_full_pipeline.py --paper-reference-csv <论文数据CSV>`。

本模块的默认结果不是实测数据，也不作为正式规划结论。阶段演化顺序只写入诊断报告，不修改计算值或阻止绘图；配置中的展示值覆盖已禁用。

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
