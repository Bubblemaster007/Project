# 课题2全流程联调项目

本项目把现有第2章源荷动态预测代码、极端风光荷场景生成包、第3章故障耦合代码、第4章电力电量平衡分析、第5章策略触发和第6章源储规划串成一个可运行流程。

当前网架拓扑先使用 `兰考算例数据.xlsx` 代替真实拓扑。程序会读取其中的 `线路`、`节点` 工作表，生成第3章故障模型需要的 `device_params.csv`，并把线路、主变、储能、应急电源、外部通道和新能源汇集设备纳入故障概率与可用容量计算。

## 目录说明

- `run_full_pipeline.py`：一键运行第2章到第6章。
- `config.yaml`：路径、近中远期参数、兰考拓扑派生参数、平衡分析资源参数、策略阈值和规划参数。
- `src/chapter3/`：极端场景 adapter、兰考拓扑 adapter。
- `src/chapter4/`：电力电量平衡分析 adapter 与可运行实现。
- `topic2_remaining_code/`：已有第3章、第5章、第6章核心模块，保持原算法逻辑。
- `ExtremeScene_extreme_generation_package_clean_20260706_133018/`：已有极端风光荷生成代码包，训练/生成算法保留；demo 流程通过 adapter 生成统一字段的 36 h 极端场景。
- `scripts/`：分章运行脚本。
- `tests/`：字段、章节和端到端测试。

## 快速运行

```powershell
python run_full_pipeline.py --config config.yaml --demo
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
