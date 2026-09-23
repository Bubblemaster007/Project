# 课题2全流程联调项目

本项目把当前论文主方法（SC-RCRB 极端场景生成、风险/结构校准、历史日历年度嵌入、局部过渡认证和全局位置分配）与第3章设备故障、第4章电力电量平衡、第5章策略触发和第6章源储配置演示串成一个可运行流程。

## 当前进度（2026-09-23）

- 已下载并转换 MATPOWER `case33bw`，形成 33 节点、32 条径向支路的测试网络；基准潮流最低电压约 0.9598 p.u.、估算网损约 186.3 kW。
- 已接入 `/Users/bubble/Code_Paper/ExtremeScene` 的当前论文主方法：SC-RCRB 条件残差生成、风险与结构校准、历史日历采样、局部 LP 认证和 `adaptive_fair_transport` 全局位置分配。
- 已在参考 `singleton` 数据上重新生成并嵌入 2021 年年度序列：72 个历史请求成功嵌入 69 个，3 个失败请求保留在事件级结果中；因此当前结果不是“全部请求完整嵌入”。
- 已将认证后的年度源荷序列接入设备故障概率、故障容量修正、逐时电力电量平衡、策略触发和配置建议模块。
- 论文仓库的年度日历/位置分配/局部过渡测试共 21 项通过。阿拉山口天气、源荷和设备参数尚未接入；当前结果属于方法联调和 IEEE 33 节点验证，不是阿拉山口正式结论。
- 当前下游仍是验证版：设备状态为示范性故障抽样，平衡采用汇总容量模型，配置采用启发式建议；节点级故障潮流、完整故障—修复过程和机会约束配置尚未完成。

当前论文方法配置和运行结果见：`configs/current_paper_case33.json`、`outputs/current_paper/singleton/2021/seed42/`。来源、参数和代码哈希记录在该目录的 `provenance.json`，进一步说明见 `临时文件/当前论文方法替换说明.md`（该文件在项目外部，不影响运行）。

默认 `config.yaml` 使用 IEEE 33 节点测试网络；兰考 Excel 仍作为兼容输入保留。真实阿拉山口数据到位后，应替换拓扑、源荷、天气和设备参数，并重新校准脆弱性曲线。

## 目录说明

- `run_full_pipeline.py`：一键运行第2章到第6章。
- `config.yaml`：路径、近中远期参数、兰考拓扑派生参数、平衡分析资源参数、策略阈值和规划参数。
- `src/chapter3/`：极端场景 adapter、兰考拓扑 adapter。
- `src/chapter4/`：电力电量平衡分析 adapter 与可运行实现。
- `phase_balance_analysis/`：承接第四章逐时调度结果，按灾前准备、灾害冲击、灾害持续、灾后恢复四阶段计算六项概率性平衡指标，并输出测试报告与 PNG/SVG/PDF 图表。
- `topic2_remaining_code/`：已有第3章、第5章、第6章核心模块，保持原算法逻辑。
- `ExtremeScene_extreme_generation_package_clean_20260706_133018/`：已有极端风光荷生成代码包，训练/生成算法保留；demo 流程通过 adapter 生成统一字段的 36 h 极端场景。
- `data/case33/`：MATPOWER `case33bw.m` 及转换后的 `nodes.csv`、`lines.csv`。
- `scripts/run_current_paper_method.py`：调用 Code_Paper 当前 SC-RCRB 与年度嵌入方法。
- `src/chapter3/current_paper_adapter.py`：把论文年度序列接入课题2下游流程。
- `scripts/`：分章运行脚本。
- `tests/`：字段、章节和端到端测试。

## 快速运行

```powershell
python run_full_pipeline.py --config config.yaml --demo
```

运行当前论文主方法和 IEEE 33 节点验证：

```bash
/Users/bubble/Code_Paper/.venv/bin/python run_full_pipeline.py \
  --config configs/current_paper_case33.json
```

分步运行：

```powershell
python scripts/make_demo_data.py --config config.yaml --output data/demo
python scripts/run_chapter2.py
python scripts/run_chapter3.py
python scripts/run_chapter4.py
python scripts/run_chapter5_6.py
```

运行测试：

```powershell
python -m pytest tests -q
```

运行兰考算例的极端事件全过程分阶段平衡分析：

```powershell
cd phase_balance_analysis
python run_phase_balance_analysis.py --config config/phase_balance_config.yaml
```

## 关键输入字段

历史源荷数据至少包含：

| 字段 | 含义 |
|---|---|
| `timestamp` | 时间戳 |
| `load_kw` | 负荷功率 |
| `wind_kw` | 风电出力 |
| `pv_kw` | 光伏出力 |

可选字段包括 `temperature`、`wind_speed`、`irradiance`、`rainfall`、`icing`、`dust`，用于构造极端天气和故障耦合输入。

## 主要输出

- `outputs/chapter2/load_probability_boundary.csv`
- `outputs/chapter2/renewable_probability_boundary.csv`
- `outputs/chapter2/source_load_features.csv`
- `outputs/chapter2/extreme_window_candidates.csv`
- `outputs/chapter2/annual_background_near.csv`
- `outputs/chapter2/annual_background_mid.csv`
- `outputs/chapter2/annual_background_far.csv`
- `outputs/chapter3/extreme_36h.csv`
- `outputs/chapter3/hazard_36h.csv`
- `outputs/chapter3/device_params.csv`
- `outputs/chapter3/device_state_sequence_36h.csv`
- `outputs/chapter3/coupled_condition_36h.csv`
- `outputs/chapter3/annual_random_production_sequence.csv`
- `outputs/chapter4/metrics.json`
- `outputs/chapter4/hourly_balance.csv`
- `outputs/chapter5/strategy_trigger_result.csv`
- `outputs/chapter6/planning_result.json`
- `outputs/chapter6/planning_result.csv`
- `outputs/run_summary.json`

## 字段与公式变量对应

| 报告变量 | 代码字段 |
|---|---|
| `L_t` | `load_kw` |
| `P_{w,t}` | `wind_kw` |
| `P_{pv,t}` | `pv_kw` |
| `N_t` | `net_load_kw` |
| `B_t` | `source_load_deviation_kw` |
| `M_t` | `source_load_match_rate` |
| `m_t` | `source_load_mismatch_flag` |
| `P_i^f` | `fault_prob` |
| `S_{i,t}` | `state` |
| `P_i^{avail}` | `available_kw` |
| `P_{s,t}^{def}` | `power_deficit_kw` |
| `P_{s,t}^{cur}` | `curtailment_kw` |
| `SOC` | `soc` |

## 替换真实算法的位置

- 若已有极端生成模型已输出 `extreme_36h.csv`，在 `config.yaml` 中设置 `extreme_generator.external_output_file`，adapter 会统一字段后继续第3章。
- 若要使用真实历史源荷数据，将 `chapter2.historical_csv` 指向真实 CSV，并取消 `--demo`。
- 若后续有真实网架拓扑，把 `topology.lankao_excel` 替换为新的拓扑文件，并在 `src/chapter3/lankao_topology_adapter.py` 中补对应 parser。

所有 CSV 均使用 `utf-8-sig` 保存，便于中文 Windows 环境直接打开。
