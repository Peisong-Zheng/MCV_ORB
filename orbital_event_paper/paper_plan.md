# 新版论文写作与图件计划

2026-09-09。原写作计划已落实为 `main.tex` / `SI.tex` 的完整英文初稿。本文保留章节论证和图表设计的依据；三个新增正文图已生成；本轮精简为一个 MIS6 事件年代表，并合并两组补充图，具体构建方式见 README。

## 1. 论文要回答的核心问题

**在控制近期事件历史和背景气候后，岁差相位是否仍与已发表增暖事件的发生率相关？这种关联有多稳健，效应大小与峰值又有多精确？**

推荐标题：**Precession-phase dependence of abrupt warming occurrence in Greenland and speleothem records**。
备选更谨慎标题：*Precession phase and abrupt warming occurrence beyond background climate variability*。

叙事顺序为：数据与观测范围 → 条件相位关联 → 关联检验与效应精度 → 其他轨道表述和物理解释。
主证据是 34 个 NGRIP 增暖开始与 21 个石笋转折组成的 55 事件目录；Barker 的 70 个重建事件是单独的补充对照，不并入目录、不合并 p，也不称为完全独立重复。

推荐按此前 GRL 图形目标先做紧凑研究短文，而非把全部实验等量堆进正文。AGU 当前规则为 GRL 最多 12 publication units；1 PU 对应 500 字或一张图/表，摘要少于 150 词。按四张正文图、无正文表，暂预留约 3,600–3,900 个计入 PU 的英文词（含摘要、正文和图注）；最终用投稿时规则核算。[AGU text requirements](https://www.agu.org/publications/authors/journals/text-graphics-requirements)、[submission checklist](https://www.agu.org/publications/authors/journals/submission-checklists)。这是组织稿件的工作假设，不表示已经选定投稿期刊。

## 2. 数据叙述必须先固定的边界

| 对象 | 本稿的准确表述 |
|---|---|
| NGRIP | 34 个已发表 GI starts；观测 12–123 kyr BP，响应 12–121.5 kyr BP |
| 石笋合成段 | MF 16 个 + Sofular 5 个已发表转折；本研究在发表标签附近赋予可复现的数值事件时间，没有重新发现 21 个气候事件 |
| 项目 `MIS6` 名称 | 观测 132.5–204.5 kyr BP，响应 132.5–203 kyr BP；年龄窗超出严格 MIS 6，稿中称 **MIS 6-centered speleothem sequence** 并明确范围；文件名保留 |
| 合成支持 | 两段的间隙不提供观测时间，也不传递历史；901 个响应 bins、180 kyr 暴露、55 个事件 |
| Barker | Table S3 的 SpeleoAge，variable threshold 70 事件；观测 0–400 kyr，响应 0–398.5 kyr；fixed threshold 59 事件仅作点年代定义敏感性 |
| 年代零点 | 计算统一 BP1950；NGRIP 与天文输入各按来源转换一次；Barker SpeleoAge 原始参考年仍属工作假设，不写成已完全核实 |
| 研究对象 | 单位整个观测时间内的增暖开始发生率，不是仅在冷态内的触发危险率，不把无事件时段删掉 |
| 主要检验 | 背景模型包括历史、LR04、CO2（合成段另含段截距），再检验新增岁差 sine/cosine；未重新正式校准“背景变量本身是否显著”这一独立问题 |
| 术语 | PI = 条件拟合对数似然改善 / (响应事件数 × ln2)，单位 bits/event；不是已经验证的样本外预测能力或因果信息流 |

支持范围选择属于本研究的分析决定，不能归称为原文明确给出的事件检测边界。LR04 是混合深海温度与冰量影响的底栖同位素背景指标，不能直接写成纯冰量。

## 3. 正文按章节写什么

下列 BibTeX keys 已在共享 `reference.bib` 中保留或补齐；详细核验来源见 `reference_audit.md`。正文通常每个论证选最直接的 1–3 篇，不把表中全部文献一次堆入同一句。

### Front matter：标题、Key Points、摘要、Plain Language Summary

- Key points 预留三条：条件相位关联；年代稳健性与效应精度不同；相关轨道指标和年代依赖限制唯一机制解释。
- 摘要最后写，按“问题 → 55 事件设计 → PI/经验 p 与补充记录 → 谨慎物理意义”的顺序，优先给主目录 PI=0.179、bootstrap p=0.0022；Barker 用一句补充。
- 摘要不罗列全部模型或敏感性 p，不放文献，不写“证明岁差驱动/精准预测事件”。
- 通俗摘要解释：研究的是某些轨道阶段是否更容易发生突变，不是每次事件的确定性时钟。

### 1 Introduction（约 500–600 词，三至四段）

1. **为什么关注单个增暖开始。** 从 D–O 型快速变化及跨区域表达切入，界定事件时间与背景状态的区别。
   引用 `dansgaard1993evidence`、`rasmussen2014stratigraphic`、`corrick2020synchronous`。
2. **已有工作知道什么。** 背景气候可改变突变/振荡的易发性；轨道尺度可调制千年变率幅度。
   引用 `barker2011800`、`sun2021persistent`、`hodell20231`，并重点讨论 `thirumalai2020methane` 的甲烷与中国石笋变率幅度对比；机制上下文选 `vettoretti2022atmospheric` / `zhang2021direct`。
3. **明确知识缺口。** 变率振幅有岁差带信号，不等于个别事件发生时间服从岁差；内部随机过程也能产生突变。
   引用 `ditlevsen2005recurrence`、`kleppin2015stochastic`，并加入 `lohmann2018random` 的冷暖态停留时间研究和 Fohlmeister (2023) 的间冰阶发生研究。Hodell (2023) 也统计事件数，不能笼统称已有研究只研究振幅。
4. **提出当前检验。** 已发表 Greenland + 石笋目录、条件发生率、年代及抽样检验、Barker 补充。
   引用 `fohlmeister2023role`、`held2024dansgaard`、`rasmussen2014stratigraphic`、`barker2011800`；首次引用 **Fig. 1**。

旧版 Introduction 的前三类背景论述可重写复用；介绍 Rousseau 弱/强季风目录的两段全部替换。

### 2 Data and methods（约 850–1,000 词）

**2.1 Published event inventories and observation support。** 说明来源、MF/Sofular 接合与重复事件处理、数值时刻定位、时间范围、空白区间和段截距。
引文：`ngrip2004high`、`rasmussen2014stratigraphic`、`fohlmeister2023role`、`held2024dansgaard`、`barker2011800`；**Fig. 1、Text S1、Figs. S1–S2、Table S1**。
Barker 年代来源直接引其原文/SOM，必要时引 `wang2008millennial`、`cheng2009ice`；不把 2016 年 Cheng 复合记录误写成 Barker 2011 的原始定年依据。

**2.2 Forcings, chronology convention and phase。** LR04、CO2、La2004；固定强迫；相位由极值构造，最低值0°、最高值180°，朝更老年代递增；标量变量按响应范围减均值/除极差。
引文：`lisiecki2005pliocene`、`bereiter2015revision`、`laskar2004long`；相位方法是本研究实现，圆周统计背景可引 `mardia2000directional`。`inso`/`paillard2026insolation` 只在真实输入生成过程能确认时用于来源，不作为当前拟合的默认依赖。正文简述，年代审计放 **Text S2**。

**2.3 Conditional event-rate model and information。** 给出两条主要公式即可：

\[
\log\lambda_i=\beta_0+\beta_H H_i+\beta_L L_i+\beta_C C_i+\beta_S S_i+\beta_s\sin\phi_i+\beta_c\cos\phi_i,
\quad Y_i\sim\operatorname{Poisson}(\Delta t_i\lambda_i),
\]
\[
I=\frac{\ell_{\mathrm{full}}-\ell_{\mathrm{BG}}}{N\ln2},\qquad LR=2(\ell_{\mathrm{full}}-\ell_{\mathrm{BG}}).
\]

BG 为删去两个相位项的模型；Barker 不含段对比。0.2-kyr bins、1.5-kyr older-bin 历史、当前 bin 排除、真实 bin 暴露作为 offset。峰值 atan2 和最大/最小率比 exp(2 hypot) 可放 SI。
引文：`davis2003observation`、`truccolo2005point`，实现引用 `virtanen2020scipy`。`kim2011granger` 可作似然比较方法背景，但本稿不称 Granger 因果检验。**Text S3、Fig. S8**。

**2.4 Calibration, uncertainty and sensitivity。** 正文只区分四件事：reduced-null bootstrap 检验关联、年代 MC 检查定年稳健性、full-model bootstrap 估计效应精度、替代模型检查设定敏感性。两目录各9,999次 reduced bootstrap。完整模型模拟与三类效应区间目前只用于主目录，不能替 Barker 虚构相同分析。
引文：`davison1997bootstrap`；NGRIP 年代引 `rasmussen2022gicc05`、`svensson2008sixty`、`moseley2020nalps19`、`corrick2020synchronous`；石笋引源研究。
所有复杂误差构造放 **Texts S2–S5**；简述其他轨道与 LR04 交互设计，见 **Texts S6–S7**。

### 3 Results（约 800–950 词）

**3.1 Conditional precession association（Fig. 2）。** 主结果：55事件，PI=0.17933，LR=13.6732，bootstrap p=0.0022；点峰值330.00°，最大/最小率比4.94。Rayleigh p=0.05796 是描述性背景，不用其是否显著来选择主假设。对照两者回答的问题，不能把“无条件不显著、条件显著”单独解释成具体机制已被识别。

**3.2 Chronology robustness and effect precision（Fig. 3）。** 10,000 套联合年代中9,982有效；PI中位0.16215，95%年代范围0.08616–0.21746；其中98.69%名义p<0.05，这是稳健性比例。抽样下相位约286.53°–374.08°、率比1.56–15.58，说明效应并不精确。A/C工作范围与B联合区域投影不同，不能把三条横线统称95%CI，不能要求C必然包住B。

**3.3 Barker supporting comparison（Fig. 2，详见 Figs. S5、S7、S8）。** 主定义70事件，PI=0.08936，bootstrap p=0.0138，峰值336.91°、率比2.885；fixed59事件 PI=0.10423、峰值350.33°，仅点年代。用“相位响应大体一致”的补充语气，不称独立年代复制或更优检测法；不拿两目录原始 logL/p 比高低。

**3.4 Orbital alternatives and model robustness（Fig. 4）。** BG/Pre/Orb 三种嵌套比较分开解释。夏至日照与相位明显共享信息：控制日照后相位PI降至0.07870/0.03830；不能排除日照、偏心率或倾角的间接路径。额外九项比较的名义 Holm 校正与原主检验分开。设计网格、背景多项式、事件间隔记忆和分段相位检查用一段归纳，明细转 **Figs. S9–S10、Texts S6–S7**。`holm1979simple` 引于多重比较方法处。

### 4 Discussion（约 650–800 词）

1. **从条件关联到物理解释的距离。** 相位接近岁差指数最小值，与季节辐射相关的易发性变化相容；不把相位差直接换成物理lag，不把概率起伏写成必然触发。引用 `zhang2021direct`、`dokken2013dansgaard`、`pedro2022dansgaard`、`vettoretti2022atmospheric`，从中选直接相关机制，别堆机制清单。
2. **轨道指标信息重叠。** 夏至日照本身由轨道分量组成，模型比较未给出唯一外部机制。背景控制可能吸收间接轨道作用，不能从额外信息弱推断物理作用不存在。
3. **背景是否调节岁差（Fig. S11）。** 预选 LR04 的两项交互；p=0.373/0.0657，Barker曲线有提示但证据不足。高LR04点拟合起伏更强不能独立证明交互，年代PI区间大于零也不代表显著。此结果用于讨论，不升为论文主结论。
4. **局限与适用范围。** 事件少、库存完整性和支持边界为条件、全时段率不是冷态危险率、强迫年代固定、年代模型简化。NGRIP modeled-extension 的轨道约束以及 Barker 石笋匹配使独立性有限，随机扰动不能自动消除结构依赖。引用 `wolff2010millennial`、`southon2004radiocarbon`、`barker2011800`；精确主张逐条对照原文，避免把“存在轨道约束”夸大为全部事件循环定年。

### 5 Conclusions（约 100–140 词）

只收束三点：岁差提供背景/历史之外的条件信息；关联较稳健但强度与峰值精度有限；物理途径及年代独立性尚不能唯一确定。不复述全套检验数字。

### Open Research / acknowledgements / author statements

- 数据：事件目录、支持区间、处理后的驱动、MC结果、来源 DOI；区分原数据与派生产品。
- 代码：主模型和随机种子、分析版本、同步映射与论文构建方式；投稿前再创建真实仓库/数据归档 DOI，不能沿用旧版URL或造占位 DOI。
- 作者、贡献、资助、利益冲突均保留待填写，不推断姓名或经费。源数据许可/再分发条件在最终归档时逐项核对。

## 4. 正文图安排与制作状态

| 编号 | 要让读者看懂什么 | 具体面板与来源 | 当前状态 |
|---|---|---|---|
| Fig. 1 | 事件的时间位置、天文与背景输入、目录观察区间 | 400→0 kyr BP 四层概览：(a) 偏心率、倾角、65°N 夏至日照量，分别使用原始单位的三条纵轴；(b) 岁差曲线上叠加 NGRIP–MIS6、Barker 变阈值和固定阈值事件，保留主目录支持条带；(c) LR04（纵轴反向）；(d) CO2。共有事件不抖动，空心方形标识固定阈值子集 | **已完成**：`figures/paper_summary/data_overview.pdf` |
| Fig. 2 | 加入岁差如何改变时间域拟合；无条件相位分布与条件效应的区别 | 六面板，上组 NGRIP–MIS6、下组 Barker：(a,d) full−reduced 拟合发生率差值、零参考线及同一坐标内底部事件标记；(b,e) 20° 扇区相位计数，隐藏径向计数 2 的文字；(c,f) 恢复标签式 PI、nominal LR p、偏好相位及最大/最小率比，在上方预留空间避免遮挡。统一主目录蓝、Barker 变阈值玫瑰、固定阈值绿。差值不解释为局部 PI 或纯岁差项；bootstrap 留在正文及 S8 | **已完成重绘**：`figures/paper_summary/phase_comparison.pdf`，读取已保存拟合，未重新拟合 |
| Fig. 3 | 关联检出不代表峰值/强度估计精确 | 使用现有四面板效应图：相位、率比、联合系数区域、响应曲线；图注严格区分A/B/C的区间含义。终稿若过密可把系数椭圆移入SI，但不能在本轮悄悄改变区间定义 | **可直接使用**：`NGRIP_MIS6_effect_uncertainty.pdf` |
| Fig. 4 | 其他轨道指标增加多少信息、和相位有多少重叠 | 上行NGRIP–MIS6、下行Barker，各3列同尺度；原BG/Pre/Orb紧凑布局；重新组合时面板统一(a)–(f)，不另加大段图内解释 | **已完成组合**：`figures/paper_summary/orbital_comparison.pdf`，读取两套已保存比较结果 |

三个正文组合图已由 `paper_summary_figures.py` 读取保存结果绘制，并接入自动同步；没有重新拟合。Fig.2 使用原始相位计数，省去任意缩放的 Rayleigh 向量和显著性圆环。主定义的经验 p 来源明确，fixed 或其他模型不共享该检验。

图件生产输入：Fig.1 读取合成和Barker主目录、观测区间、两个主分析的 binned inputs（驱动曲线用完整Barker支持，同期一致性核对）；Fig.2 读取事件相位、主分析拟合系数/summary，以及已保存的分箱发生率。NGRIP–MIS6 的 B/BP 发生率取自参数对齐的 orbital-driver 实验，逐项核对主分析系数、事件数、有效时长及 LR；Barker 直接读取主分析的 reduced/full 发生率。Fig.4 读取两套 orbital `comparison_summary.csv`。原始PDF不裁成位图、不修改原分析输出。所有图最终保存矢量PDF及PNG预览。

## 5. Supporting Information 结构

| 文本 | 内容 | 图／表及主要引文 |
|---|---|---|
| Text S1 | 已发表事件、MIS6-centered接合、锚点/梯度、数据覆盖与无事件尾段 | Figs. S1–S2，Table S1；Rasmussen2014、NGRIP2004、Fohlmeister2023、Held2024、Barker2011 |
| Text S2 | 数据年代零点和三类工作年代模型 | Figs. S3–S5；Rasmussen2022 dataset、Svensson2008、Moseley2020、Corrick2020、源石笋研究、Barker2011 |
| Text S3 | 分箱、older-bin历史、offset、相位、PI、拟合诊断、reduced bootstrap | Fig. S8；Davis2003、Truccolo2005、Mardia2000、Davison1997、SciPy2020 |
| Text S4 | 年代误差对PI、峰值与强度的影响 | Figs. S6–S7；有效分母、边界剔除、不得补抽，So-57替代和NGRIP网格敏感性在文字中说明 |
| Text S5 | 主目录full-model抽样：A/B/C与相位展开、联合区域、误差来源差异 | 正文Fig.3；不添加Barker尚未做的full-model精度分析；Davison1997 |
| Text S6 | 分箱/历史网格、背景形状、等待时间、分段相位与Barker定义 | Figs. S9–S10、正文 Fig. 2e,f；明确共同支持173/164.8与主180kyr的区别 |
| Text S7 | 其他驱动的十行比较／九项检验族、共线性、LR04调节 | 正文Fig.4、Fig. S11；Laskar2004、Holm1979，交互仅nominal |

当前 SI 图编号：S1 MF–Sofular重叠；S2 MIS6合成定位；S3 NGRIP年代；S4 MIS6年代；S5 Barker年代；S6 合成年代PI；S7 Barker年代PI；S8 两套bootstrap左右对比；S9 设计敏感性；S10 气候形状/记忆；S11 两套LR04交互上下对比。原 Barker 变量/固定阈值单独主图不再重复收入 SI，正文 Fig. 1 与 Fig. 2 已提供所需概览和定义敏感性对比，原始分析图保留。图号仅由论文清单管理，拼接脚本使用描述性名称。

仅保留 **Table S1：21 个 MIS6 石笋事件的数值年代**，包括合成事件标签、源记录、原文标签及本研究赋予的年代。由 `paper_tables.py` 导出 TeX、CSV 和输入哈希。其余已发表年代、单位审计、误差设置、主结果、bootstrap 及敏感性汇总不再制表；必要方法和关键结果放在相应文字与图注中。完整研究 CSV 仍保存在各分析目录，供代码与数据公开时查阅。

StalAge/iscam 文献用于解释原始年龄模型来源（`scholz2011stalage`、`fohlmeister2012statistical`），不能把本项目年龄warp说成运行了这些算法或还原其后验。`myrvollnilsen2022comprehensive`仅用于讨论另一种年代不确定性方法，不能标成本项目实际采样方法。

## 6. 旧稿复用清单

| 旧稿部分 | 操作 |
|---|---|
| 突变、背景易发性、振幅vs事件时间的动机 | 保留思路、压缩重写，核对每句引文是否直接支持 |
| 条件Poisson、offset、sine/cosine与峰值率比公式 | 改为当前历史1.5kyr、段截距、无resolution、真实支持；删多余模型序列 |
| reduced-model bootstrap动态更新历史说明 | 复用方法骨架，换当前两套9999次结果，勿与full bootstrap混淆 |
| 背景—轨道—随机机制讨论 | 保留作假说框架，删“独立验证”和确定性/样本外含义 |
| 39条旧bib | 复制保留原key，补当前源数据/年代/方法；Rousseau仍可留库但不进入当前数据引用 |
| 标题摘要与Results | 完全重写；不能沿用196事件/640kyr/两类季风反相或旧p |
| lag、KS窗口、resolution、三套Barker时间轴 | 删除当前稿中的实验叙述与图引用，不迁移旧结果 |
| 图片同步和扁平figures目录 | 保留编号映射逻辑，换白名单为当前PDF；新图缺失明确标记，不删除未列文件 |

## 7. 目录、同步与实施顺序

沿用旧版 `orbital_event_paper/main.tex`、`SI.tex`、`reference.bib`、`figures/Fig*.pdf`。新增本计划、引用审计、图件清单和简短README/Makefile。正文采用旧版 article+natbib/AGU bibliography 作为可编辑初稿，投稿前再切换官方模板。

已完成：参考文献核验；正文和 SI 全文；三个正文组合图；一个 MIS6 年代表；全部十五图同步。`make pdf` 先基于当前保存结果更新组合图和表格，再同步并编译；不重算科学模型，不从旧项目读取研究结果。

下一步是作者核稿与修订：检查主叙事、物理解释和术语；提供作者、单位、资助等真实信息；确定投稿模板并建立公开归档。当前初稿不追加 lag、逐周期、fixed-bootstrap 等新实验。初稿实施见 `docs/plans/2026-09-09-manuscript-draft.md`；图表精简见 `docs/plans/2026-09-09-si-consolidation.md`。
