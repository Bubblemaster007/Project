# 课题2全流程联调项目

本项目将论文源荷数据与第3章设备故障、第4章电力电量平衡、第5章策略触发和第6章源储配置演示连接，用 IEEE 33 节点算例验证下游流程。论文 SC-RCRB 生成与年度嵌入的正式入口仍需外部冻结文件。

## 当前进度（2026-09-23）

- 已下载并转换 MATPOWER `case33bw`，形成 33 节点、32 条径向支路的测试网络；基准潮流最低电压约 0.9598 p.u.、估算网损约 186.3 kW。
- 已使用论文原始 `singleton` FY2021 逐时风光荷与天气序列，跑通全年连续支路故障修复、33 节点线性配电调度、交流网损迭代回填与逐时电压复核，以及三个风速暴露窗口各 30 个条件设备样本的阶段分析。
- 已接通论文格式年度序列的下游入口，并对论文冻结生成文件增加预检。本机现有论文仓库缺少冻结模型、候选库和 `publication_protocol`，所以尚未重跑 SC-RCRB 生成与认证；历史交接中的 72/69/3 结果不是本机本次复现结果。
- 可用容量字段可逐时约束上级电网、变压器、应急机组、储能和风光，显式零值会按零容量处理；失联节点需求计入原始需求缺供。若年度 CSV 提供资源的逐时失效率和修复时长，还会生成跨小时连续的设备故障状态，并在条件样本中分别抽样。
- 本机参考数据运行 8760 小时、27 次线路故障起始，年度单样本缺供 90314.97 kWh、失负荷 323 小时；交流复核最低电压 0.95386 p.u.，无电压或上级容量越限，物理功率账本最大残差约 `6.93e-11` kW。33 项测试通过。上述数字只用于联调，不能解释为已校准的阿拉山口风险。
- 故障率、修复时长、统一支路容量、资源接入位置仍为联调假设；当前参考数据仅模拟支路故障，其他设备的条件失效率须由外部输入。
- 阿拉山口数据尚未接入；第6章目前为启发式演示，尚未完成机会约束配置优化。

本次联调结果见 `outputs/paper_reference_case33/`；修订的技术路线在项目外部的 `C:\Users\13411\Desktop\ProjectCode\临时文件\课题2技术路线_33节点论文数据联调_20260923.md`。

默认 `config.yaml` 使用 IEEE 33 节点测试网络；兰考 Excel 仍作为兼容输入保留。真实阿拉山口数据到位后，应替换拓扑、源荷、天气和设备参数，并重新校准脆弱性曲线。

## 目录说明

- `run_full_pipeline.py`：根据入口运行演示流程、论文原始数据 33 节点联调或论文年度 CSV 下游分析。
- `config.yaml`：路径、近中远期参数、兰考拓扑派生参数、平衡分析资源参数、策略阈值和规划参数。
- `src/chapter3/`：论文年度序列 adapter、拓扑 adapter、可选资源序贯故障修复。
- `src/chapter4/`：IEEE 33 节点逐时调度、交流潮流复核与电力电量指标。
- `phase_balance_analysis/`：独立的合成典型过程绘图和指标演示；论文参考数据的正式阶段统计在 `outputs/paper_reference_case33/`。
- `topic2_remaining_code/`：已有第3章、第5章、第6章核心模块，保持原算法逻辑。
- `ExtremeScene_extreme_generation_package_clean_20260706_133018/`：已有极端风光荷生成代码包，训练/生成算法保留；demo 流程通过 adapter 生成统一字段的 36 h 极端场景。
- `data/case33/`：MATPOWER `case33bw.m` 及转换后的 `nodes.csv`、`lines.csv`。
- `scripts/run_current_paper_method.py`：调用 Code_Paper 当前 SC-RCRB 与年度嵌入方法。
- `src/chapter3/current_paper_adapter.py`：把论文年度序列接入课题2下游流程。
- `scripts/`：分章运行脚本。
- `tests/`：字段、章节和端到端测试。

## 快速运行

论文原始 `singleton` 风光荷与 IEEE 33 节点联调（需先安装 `requirements.txt`，本机验证版本见 `requirements-tested.txt`）：

```powershell
python run_full_pipeline.py --paper-reference-csv <论文FY2021逐时CSV的绝对路径> --event-samples 30
```

此入口执行全年连续故障修复、节点级线性配电调度、逐时交流潮流网损回填与电压复核，以及按天气选出的三个风速暴露窗口的条件多样本阶段平衡；结果在 `outputs/paper_reference_case33/`。网络和资源参数来自 `config.yaml` 的 `chapter4`、`topology`、`case33_network`，其中支路额定容量、设备故障率和修复时长仍为联调假设，相关概率只条件于参考源荷天气轨迹。下述 `current_paper` 入口还依赖外部论文仓库中的冻结生成模型、候选库和 `publication_protocol`；克隆本项目不能单独复现该入口。可用 `CODE_PAPER_ROOT` 环境变量指向完整论文仓库、`PAPER_SOURCE_CSV` 指向跨机器迁移后的原始 CSV，入口会先检查所需文件。其下游已改用同一套 33 节点调度，但参考日历天气与生成风光事件的联合一致性仍待验证。

若已有论文方法导出的 `annual_source_load.csv`，可跳过生成步骤，直接运行 `python run_full_pipeline.py --paper-annual-csv <年度CSV绝对路径> --paper-provenance-json <可选的来源JSON>`。输入必须包含逐时 `timestamp`、`load_kw`、`wind_kw`、`pv_kw`、`event_id`、`wind_speed`；可选的 `available_kw_grid_channel`、`available_kw_transformer`、`available_kw_emergency_gen`、`available_kw_storage`、`available_kw_renewable` 可逐时传入设备可用容量。变压器容量与上级电网容量取较小者；显式零值生效，负值或缺失值 NaN 报错。汇总 `available_kw_line` 缺少故障支路编号，输入时会报错，须改为逐支路故障状态。生成事件的天气一致性需另行验证。该入口保留嵌入成败记录，并明确标注 36 小时事件内的阶段边界只是测试划分。

如需模拟非线路设备的连续故障，可对 `grid_channel`、`transformer`、`emergency_gen`、`storage`、`renewable` 中的任一设备类型同时提供 `failure_rate_per_hour_<类型>` 与 `repair_hours_<类型>` 两列。失效率是每小时强度，逐时转移概率为 `1-exp(-失效率)`；故障从抽中时刻持续指定修复小时数。输入缺一列、负值或 NaN 会报错。未提供失效率的设备不会凭空生成故障；有实测容量列时，故障状态会进一步把该时段容量限制为零。事件样本的随机种子及故障起始记录见 `resource_failures.csv`。这些输入需来自校准或明确标注的假设，不能把累计脆弱性概率直接当作每小时失效率。

```powershell
python run_full_pipeline.py --config config.yaml --demo
```

完整论文冻结文件到位后运行当前论文主方法和 IEEE 33 节点验证：

```powershell
python run_full_pipeline.py --config configs/current_paper_case33.json
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
python -m pytest -q --import-mode=importlib
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

新建的联调 CSV 由 pandas 以 UTF-8 保存；Windows 表格软件若错误识别中文编码，可在导入时选择 UTF-8。

## 依据与未完成项

- 33 节点数据与基值来自 [MATPOWER case33bw](https://matpower.org/docs/ref/matpower6.0/case33bw.html)；径向潮流近似参考 [Baran 与 Wu 的 DistFlow 论文](https://ecal.berkeley.edu/tbsi/Energy-Systems-Optimization-Course/References/Baran89%20-%20UCB%20-%20DistFlow.pdf)。`case33bw` 未提供可直接使用的线路热额定值，当前统一 5000 kW 仅为测试约束。
- 需要补齐论文冻结模型、候选库和 `publication_protocol`，才能重跑 SC-RCRB 场景生成与年度认证；需要联合天气轨迹、设备故障修复数据及阿拉山口实际数据，才能校准条件与年度风险。
- 需要进一步验证孤岛成网资源、线路实测容量、应急燃料和滚动调度；当前第6章规划仍是启发式演示。
