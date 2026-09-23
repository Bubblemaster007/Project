# IEEE 33 节点基准算例

来源：MATPOWER 官方仓库 `case33bw.m`（Baran–Wu 33-bus distribution system）。

`case33bw.m` 原始文件包含 33 个节点、37 条支路，其中后 5 条为常开联络支路；本项目基准径向拓扑使用前 32 条支路。`nodes.csv` 和 `lines.csv` 为脚本转换后的项目数据格式。

运行转换和基准径向潮流检查：

```bash
cd Project
/Users/bubble/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12 scripts/prepare_case33.py
```

结果保存在 `baseline_summary.json`，包括节点/支路数、总负荷、最低/最高电压及估算网损。
