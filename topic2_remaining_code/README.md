# 课题2剩余代码模块说明

本代码包用于补齐已有“极端风光荷场景生成代码”和“电力电量平衡分析代码”之外的技术链条。

## 模块

1. `fault_probability_model.py`：设备故障概率、设备状态抽样、可用容量修正。
2. `coupled_condition_builder.py`：将36 h极端风光荷场景与设备状态耦合，形成“源荷-故障耦合极端工况”。
3. `annual_sequence_embedder.py`：将极端工况嵌入8760 h常规源荷背景，形成随机生产模拟序列。
4. `strategy_trigger.py`：根据第4章六项指标和辅助变量，触发八项保供策略。
5. `resilience_reliability_planner.py`：根据策略触发结果，形成简化源储配置建议。
6. `demo_run.py`：用模拟数据跑通完整流程。

## 技术路线衔接

已有极端风光荷生成代码输出36 h源荷场景；本代码先计算设备故障概率和状态序列，再构建源荷-故障耦合极端工况，并嵌入8760 h年度背景序列。已有电力电量平衡分析代码输出六项指标后，本代码根据指标触发八项保供策略，并给出阶段性源储配置建议。

## 运行

```bash
pip install -r requirements.txt
python demo_run.py
```

## 输入字段建议

### annual_background.csv
`timestamp, load_kw, wind_kw, pv_kw`

### extreme_36h.csv
`timestamp, load_kw, wind_kw, pv_kw, event_core`

### hazard_36h.csv
`timestamp, wind_speed, icing, rain, heat`

### device_params.csv
`device_id, device_type, rated_kw, a_wind, b_wind, a_icing, b_icing, a_rain, b_rain, a_heat, b_heat, normal_factor, derate_factor, recovery_factor`

## 注意

代码变量为了可运行性保留下划线；写入技术报告时，变量应改成上下标形式。
