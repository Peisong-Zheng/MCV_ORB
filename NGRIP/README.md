# NGRIP 事件年代与误差

这里准备 Rasmussen et al. (2014) 的 69 个事件边界：34 个 GI 增暖起点、
35 个 GS 降温起点。联合研究只使用 34 个 GI；GS 保留在年代抽样中，
用于约束整个事件序列的地层顺序。

## 两步运行

在 `NGRIP/` 目录打开并依次运行 `ngrip_data_preparation.ipynb`，然后运行：

```bash
python ngrip_event_age_uncertainty.py
```

也可以从项目根目录一次执行：

```bash
cd NGRIP
jupyter nbconvert --to notebook --execute --inplace ngrip_data_preparation.ipynb
python ngrip_event_age_uncertainty.py
cd ..
```

路径都直接相对于 `NGRIP/`，不自动寻找项目根目录。Notebook 已保留执行结果，
便于逐格核验；重跑会覆盖两个处理表。抽样脚本会覆盖自己的结果表和图。

## 数据流

```text
data/raw/Rasmussen2014_GI_GS_starts_no_subevents_wide.xlsx
    └─ Notebook：读取 Long_no_subevents，准备事件年代和误差
        └─ data/processed/ngrip_warming_cooling_starts.csv

data/raw/Rasmussen et al-2022-GICC05_time_scale.txt
    └─ Notebook：核对 MCE，准备原年代网格
        └─ data/processed/ngrip_chronology_grid.csv

以上两个 processed 表
    └─ ngrip_event_age_uncertainty.py：combined 抽样 → 顺序拒绝 → 保存、画图
```

原始 Excel 和处理后的事件 CSV 已分别移到 `NGRIP/data/raw/` 与
`NGRIP/data/processed/`；根目录原位置的副本已删除。
年层 MCE TXT 位于 `NGRIP/data/raw/`，只由 Notebook 读取。

Excel 保留原来的三个工作表。宽表增加 GI/GS 原文标签、定义误差代码和 MCE；
长表增加原文标签、定义误差代码、原文解释和 PDF 页码。新增字段已核对
Rasmussen Table 2（本地 PDF 第 10 页、页内编号第 9 页）。原有事件、年代、
深度及 MCE 数值未改；PDF 提取代码和中间表只放在临时目录。
没有单独增加定义误差原始补充表。

既定目录不单独计入字母子事件；没有单独母事件行时，以最古老的子事件起点
代表母事件，例如原文 GI-1e 对应保留标签 GI-1。原文标签保留用于回查。

## Notebook 的四个代码单元

1. 读取 Excel 长表。
2. 选取需要的列，转换年代零点，准备定义误差和年代包络。
3. 读取年层 MCE，准备 5 kyr 主网格及已有敏感性实验的 2.5、10 kyr 网格。
4. 检查事件数、顺序、误差完整性及 44 个表内 MCE，保存两张处理表。

事件 CSV 有九列：

| 列 | 含义 |
|---|---|
| `event_label` | 保留的 GI/GS 标签，也是 realization 列名依据 |
| `source_event_label` | Table 2 原文标签，含必要的子事件字母 |
| `event_type` | warming / cooling |
| `age_ka_b2k` | 原始千年 b2k 年代 |
| `age_ka_bp` | 千年 BP1950 年代，等于上一列减 0.05 |
| `definition_uncertainty_code` | 原文定义误差代码或明确的 ±4 |
| `definition_sigma_yr` | 定义误差的工作标准差，年 |
| `chronology_envelope_yr` | 原文 MCE 或模型延伸段工作包络，年 |
| `chronology_source` | MCE / model_ext |

定义误差仍沿用 a=20、b=50、c=200、d=100、f=30 年的独立高斯标准差。
b 的原文范围 40–60 年取中点 50，f 的 20–40 年取 30，采用 NGRIP δ18O 的解释。
明确的 ±4 年条目沿用 σ=4 年；原表没有把这些条目标成 1σ。
高斯形状、独立性和上述数值化方式都是研究假设。20 年记录分辨率不重复相加。

网格 CSV 有 `knot_spacing_ka`、`knot_age_ka_b2k` 和
`chronology_envelope_ka` 三列。它是 Notebook 的计算结果，不是新增原始资料。

## 抽样脚本的结构

参数集中在文件开头：10,000 条、seed=20260907、5 kyr 主网格。
这个 seed 是旧流程 combined 分支的实际 seed。

- `chronology_process_basis`：把准备好的包络除以 2 作为工作 σ，构造累计
  方差增量，并把节点偏移线性插值到事件处。
- `sample_age_realizations`：抽取共享的年代增量和各事件独立的定义误差。
  先排除非单调节点年代映射，再排除 combined 事件交错；不通过排序修复。
- `realization_table`：将样本排成兼容下游读取的宽表。
- `plot_uncertainty`：上方面板显示定义误差、年代包络及 GI/GS 事件标签；
  下方面板显示 combined 曲线、逐点分位范围和中位数。
- `main`：读两张处理表，调用抽样，保存结果和图。

原模型保留精确的年层计数终点 60.202 ka b2k，MCE 为 2.611 kyr；
其后使用 ±4.5% × b2k 年龄的工作包络，约按 2σ 解释。
方差增量在年代方向累计，使附近事件的误差相关；节点间的线性插值和
顺序拒绝均与旧方案相同。事件处包络的一半与插值后实际 σ 可能不同，
两者在结果摘要中分开列出。

只运行 combined，不再生成 definition-only 或 chronology-only 抽样。
图中的抽样曲线只显示 combined。采用相同批量、随机调用顺序及 seed 后，新的
10,000 × 69 个保存年代与旧 combined CSV 逐字节一致。

## 输出与下游读取

```text
data/processed/ngrip_event_age_uncertainty/
├── ngrip_event_age_realizations.csv          # 10,000 行，ID + 69 个事件年代
├── ngrip_event_age_uncertainty_summary.csv   # 误差尺度及 combined 分位数
└── parameters_and_provenance.csv            # 参数、拒绝计数、输入和代码哈希

figures/ngrip_event_age_uncertainty/
├── ngrip_event_age_uncertainty.png
└── ngrip_event_age_uncertainty.pdf
```

Realizations 列名仍为 `age_ka_bp__GI-1` 等，所有输出年龄均为 kyr BP1950。
参数表使用 `parameter,value` 两列。主抽样接受率为 96.12%：11,000 个提案中
427 个因事件交错被拒绝，0 个因节点映射被拒绝；保留 10,000 个，另有
573 个已接受的尾部样本未使用。

`NGRIP_MIS6_event_uncertainty_sensitivity.py` 读取同一路径的 combined CSV，
按固定事件标签选择 34 个 GI，与 MIS 6 样本配对。联合目录仍位于
`data/curated/ngrip_mis6_warming_events.csv`，无需改动。
已有网格间距筛查脚本已改为读取这两张 processed 表。

本脚本不截断到下游观察窗口。高斯尾部造成的支持范围越界，仍由联合分析
按原规则处理。工作包络不是硬上下界，MC 分位范围也不是校准过的置信区间。
模型延伸段不确定性没有官方逐点定量模型；统一 4.5% 是本研究采用的情景，
Moseley et al. (2020) §5.2 的原文结论是外推到 120 ka 时约达 4.5%。
原研究对于轨道调谐、系统计数偏差、事件存在性及外部驱动年代误差的
限制仍适用，详见项目方法说明。

迁移及一致性核验见 [2026-09-07 核验记录](../docs/reviews/ngrip-simplification-2026-09-07.md)。

## 主要来源

- [Rasmussen et al. (2014), Table 2](https://doi.org/10.1016/j.quascirev.2014.09.007)
- [Rasmussen et al. (2022), GICC05 年层 MCE](https://doi.org/10.1594/PANGAEA.943193)
- [Moseley et al. (2020), §5.2](https://doi.org/10.5194/cp-16-29-2020)
