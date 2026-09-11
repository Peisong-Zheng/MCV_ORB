# 新论文参考文献核验记录

核验日期：2026-09-09。`reference.bib` 从旧项目
`/Users/pz/VScode/DO_warming/monsoon_paper/reference.bib` 复制；旧文件未修改。
保留原有 **39 个 citation keys**，增加 **18 条**，共 **57 条**。
本文件区分“书目信息已核对”和“文献确实支持某个科学陈述”；前者不能代替正式写作时对正文、图表及补充材料的阅读。

## 1. 新增文献及预定用途

新增论文/专著的作者、标题、出版年、卷页和 DOI 均与出版社或 Crossref 登记记录核对；三个数据条目与 DataCite 登记及本项目原始文件头核对。Holm (1979) 使用原文/JSTOR 稳定链接，不臆造 DOI。

| Citation key | 文献与核验来源 | 适合引用的位置 |
|---|---|---|
| `held2024dansgaard` | Held et al. (2024), *Nature Communications*, 15, 1183. [出版社](https://www.nature.com/articles/s41467-024-45507-5)，DOI `10.1038/s41467-024-45507-5` | Sofular 数据、δ¹³C 事件解释、So-4/So-57 年代与堆叠方法；末次与倒数第二次冰期的发生频率差异背景 |
| `fohlmeister2023role` | Fohlmeister et al. (2023), *Communications Earth & Environment*, 4, 245. [出版社](https://www.nature.com/articles/s43247-023-00908-0)，DOI `10.1038/s43247-023-00908-0` | MF 数据和事件标签；已有关于夏季日照与间冰阶发生关系的直接研究，是引言和讨论的关键对话对象 |
| `ngrip2004high` | North Greenland Ice Core Project members (2004), *Nature*, 431, 147–151. [出版社](https://www.nature.com/articles/nature02805)，DOI `10.1038/nature02805` | NGRIP 记录覆盖及其独立于本项目的古气候证据；123 ka 观测支持的出处 |
| `rasmussen2022gicc05` | Rasmussen, Svensson & Vinther (2022), GICC05 数据集，PANGAEA. [数据页](https://doi.pangaea.de/10.1594/PANGAEA.943193)，DOI `10.1594/PANGAEA.943193` | 实际使用的年层年代/MCE 文件，b2k 年代零点；数据可用性声明 |
| `rasmussen2023ice` | Rasmussen et al. (2023), *Earth System Science Data*, 15, 3351–3364. [出版社](https://essd.copernicus.org/articles/15/3351/2023/)，DOI `10.5194/essd-15-3351-2023` | GICC05 数据发布与年层标记的背景；与上一条数据引用配套，而不是替代它 |
| `svensson2008sixty` | Svensson et al. (2008), *Climate of the Past*, 4, 47–57. [出版社](https://cp.copernicus.org/articles/4/47/2008/)，DOI `10.5194/cp-4-47-2008` | GICC05 延伸至约60 ka、累计层计误差的来源；详细年代方法放 SI |
| `moseley2020nalps19` | Moseley et al. (2020), *Climate of the Past*, 16, 29–50. [出版社](https://cp.copernicus.org/articles/16/29/2020/)，DOI `10.5194/cp-16-29-2020` | 模型延伸区年代包络的工作假设及其依据，尤其 §5.2 |
| `scholz2011stalage` | Scholz & Hoffmann (2011), *Quaternary Geochronology*, 6, 369–382. [出版社](https://www.sciencedirect.com/science/article/pii/S1871101411000094)，DOI `10.1016/j.quageo.2011.02.002` | 已发表 Sofular 年代模型使用的 StalAge；解释本项目没有重新运行这一原始年龄模型 |
| `fohlmeister2012statistical` | Fohlmeister (2012), *Quaternary Geochronology*, 14, 48–56. [出版社](https://www.sciencedirect.com/science/article/abs/pii/S1871101412001446)，DOI `10.1016/j.quageo.2012.06.007` | iscam 同洞记录同步/堆叠方法及本项目误差扰动与原始模型的区别 |
| `wolff2010millennial` | Wolff et al. (2010), *Quaternary Science Reviews*, 29, 2828–2838. [出版社](https://www.sciencedirect.com/science/article/pii/S0277379109003588)，DOI `10.1016/j.quascirev.2009.10.013` | 格陵兰突变综述、GICC05modelext 结构；讨论年代独立性的限制 |
| `southon2004radiocarbon` | Southon (2004), *Radiocarbon*, 46, 1239–1259. [原文](https://journals.uair.arizona.edu/index.php/radiocarbon/article/download/4180/3605)，[Crossref](https://api.crossref.org/works/10.1017/S0033822200033129)，DOI `10.1017/S0033822200033129` | 旧格陵兰年代模型/轨道约束的背景；与 Wolff (2010) 配套，不能单凭该文证明本项目处理已经完全消除了调谐影响 |
| `myrvollnilsen2022comprehensive` | Myrvoll-Nilsen et al. (2022), *Climate of the Past*, 18, 1275–1294. [出版社](https://cp.copernicus.org/articles/18/1275/2022/)，DOI `10.5194/cp-18-1275-2022` | 层计误差相关性、潜在系统偏差、年代不确定性局限；本项目没有实施其完整统计框架 |
| `davison1997bootstrap` | Davison & Hinkley (1997), *Bootstrap Methods and their Application*, Cambridge University Press. [Crossref](https://api.crossref.org/works/10.1017/CBO9780511802843)，DOI `10.1017/CBO9780511802843` | reduced-model 参数 bootstrap 与 full-model 效应抽样不确定性的通用方法依据；不是本研究专属 PI 指标的来源 |
| `holm1979simple` | Holm (1979), *Scandinavian Journal of Statistics*, 6, 65–70. [JSTOR](https://www.jstor.org/stable/4615733)，[原文扫描](https://sci2s.ugr.es/keel/pdf/specific/articulo/0052_001.pdf) | 其他轨道驱动比较的 Holm 多重检验校正，适合 SI |
| `wang2008millennial` | Wang et al. (2008), *Nature*, 451, 1090–1093. [出版社](https://www.nature.com/articles/nature06692)，DOI `10.1038/nature06692` | 当代于 Barker (2011) 的中国石笋记录来源、年代继承链；如保留 Sanbao 对照图亦需要 |
| `cheng2009ice` | Cheng et al. (2009), *Science*, 326, 248–252. [出版社](https://www.science.org/doi/10.1126/science.1177840)，[Crossref](https://api.crossref.org/works/10.1126/science.1177840)，DOI `10.1126/science.1177840` | Barker 石笋时间轴的历史资料背景；具体匹配点仍应直接引用 Barker 原文/SOM |
| `held2024sofularData` | Held (2024), NOAA/WDS Sofular 数据集. [NOAA 数据页](https://www.ncei.noaa.gov/access/paleo-search/study/39063)，[DataCite](https://api.datacite.org/dois/10.25921/b84y-cm81)，DOI `10.25921/b84y-cm81` | 数据可用性；注意数据库作者字段为 Held，不能从论文作者列表自动复制全体作者 |
| `fohlmeister2023mfData` | Fohlmeister et al. (2023), NOAA/WDS Melchsee-Frutt 数据集. [NOAA 数据页](https://www.ncei.noaa.gov/access/paleo-search/study/38201)，[DataCite](https://api.datacite.org/dois/10.25921/jgzt-2n35)，DOI `10.25921/jgzt-2n35` | 数据可用性；与 MF 原始文件头的 DOI 一致 |

## 2. 复用条目的核验与修正

以下八条旧条目已在本轮与 Crossref 登记信息核对；LR04 文章编号采用出版社“如何引用”的条目，因为 Crossref 的 `article-number` 返回的是稿件编号而不是 PA1003。

| 旧 citation key | 本轮处理 | 核验链接 |
|---|---|---|
| `rasmussen2014stratigraphic` | 标题的 `abrupt climate changes` 改为原文的 `abrupt climatic changes`，其余原键保留 | [Crossref](https://api.crossref.org/works/10.1016/j.quascirev.2014.09.007) |
| `barker2011800` | 作者、年份、卷页、DOI 一致；继续作为 Barker 事件和 Table S1/S3 的直接来源 | [Crossref](https://api.crossref.org/works/10.1126/science.1203580) |
| `corrick2020synchronous` | 作者、年份、卷页、DOI 一致 | [Crossref](https://api.crossref.org/works/10.1126/science.aay5538) |
| `lisiecki2005pliocene` | 补 `PA1003`，修正标题中 δ¹⁸O 的排版 | [出版社](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2004PA001071) |
| `bereiter2015revision` | 作者、年份、卷页、DOI 一致；引用时明确本项目使用的是编译 CO₂ 数据，不能把文章标题的800–600 ka当作本研究观测时段 | [出版社](https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2014GL061957) |
| `laskar2004long` | 年份、标题、卷页、DOI 一致；是当前 La2004 轨道解来源 | [Crossref](https://api.crossref.org/works/10.1051/0004-6361:20041335) |
| `berger1978long` | 作者 André L. 的空格与期刊名中的 `the` 修正 | [Crossref](https://api.crossref.org/works/10.1175/1520-0469(1978)035%3C2362:LTVODI%3E2.0.CO;2) |
| `paillard2026insolation` | **确实已经发表**，2026-03-27，CP 22, 647–673，DOI 正确；可保留为日照定义/计算的背景文献 | [出版社](https://cp.copernicus.org/articles/22/647/2026/) |

新副本中的全部 DOI 字段统一移除 `https://doi.org/` 前缀，以便由 BibTeX 样式处理 DOI 链接。没有更换任何已有 citation key，也没有批量补造作者全名。规划阶段其余 **31 条旧条目只继承**；正式写作阶段又核验其中10条，见第6节，尚有21条继承条目未宣称逐条完成核验。

## 3. 在新论文中的使用边界

- `rousseau2023reliable` 留在文献库以保留旧项目来历，但不进入当前事件数据、结果或核心证据链。文献库不是最终引用清单，正文未引用的条目不会自动出现在正常参考文献列表中。
- 引言可复用 `dansgaard1993evidence`、`lohmann2018random`、`sun2021persistent`、`zhang2021direct`、`hodell20231` 等。事件起始发生率与千年尺度振幅/变率强度不同，引用时应说明当前研究的新增问题，不能把它们的统计量当成同一量。
- 物理讨论可择要引用 `zhang2014abrupt`、`zhang2017abrupt`、`vettoretti2022atmospheric`、`dokken2013dansgaard`、`pedro2022dansgaard`、`malmierca2023dansgaard`。本研究的统计交互不足以确认机制，这些文献提供机制背景。
- `mardia2000directional` 支持圆统计；`davis2003observation`、`truccolo2005point` 支持包含历史项及外部协变量的计数/点过程建模。`kim2011granger` 只有在解释似然信息增益与相关方法的联系时才需要，不能借其标题把本项目的条件关联称为因果证明。
- `moseley2020nalps19` 支持外推到约120 ka时达到约4.5%的做法；把整个模型延伸段都赋予年龄的±4.5%是**本项目的工作情景**。不要写成 Moseley 提供了一条逐点4.5%误差曲线。
- `scholz2011stalage` 和 `fohlmeister2012statistical` 是已发表原始年龄模型的方法来源。本项目在参考年代上施加扰动，不是重新获得 StalAge/iscam 的完整后验。
- `cheng2016asian` 可以用于后续中国石笋背景或对照图；它晚于 Barker (2011)，不能成为 Barker 原有 SpeleoAge 轴的原始资料出处。
- 当前轨道输入是 `laskar2004long` 的 La2004。`paillard2026insolation` 与 `berger1991insolation` 是否进入正文，取决于是否实际讨论/使用它们的方法，不能仅因旧文引用过就将其列为当前原始数据来源。
- `lisiecki2005pliocene` 明确使用轨道调谐的年代。论文可以把 LR04 作为固定背景协变量，但不能宣称所有背景时间轴均独立于轨道信息。

## 4. 留待正式写作时核实的事项

1. Barker `SpeleoAge` 的确切年代参考年仍是项目已记录的待核事项；现流程按 BP1950 使用。查到 Wang/Cheng 的来源信息并不自动证明 Barker 输出列的零点。
2. NOAA 数据引用中的作者、发表年与研究论文可能不同，本次采用数据登记字段；正式数据可用性声明还应补最终归档地址、版本和访问日期。
3. 若使用四记录背景图（Sanbao、Huagapo、Sofular、MF），需另外核对 Burns et al. (2019) 的 Huagapo 数据引用。本轮未将未核验的该条目写进 `.bib`；当前主分析不使用 Huagapo 事件。
4. MIS6 工作目录名不等于所有事件严格落在通常定义的 MIS6 内：最老事件约194 ka。图注和数据章节需准确给出实际时段，避免仅凭文件夹名定义地层范围。

## 5. 验证

- 57 个 citation keys 唯一，旧版本的39个均保留。
- 使用 TeX Live 的 `bibtex` 对全部条目运行 `plainnat`；无 BibTeX 错误或警告。
- 临时 `\nocite{*}` 文档完成 pdfLaTeX 编译，验证重音字符、数学符号和条目类型；验证文件放在 `/private/tmp/mcv_orb_bib_validation`，未混入论文正文。
- 未调用收费学术接口；本轮检索使用出版社、Crossref、DataCite 和数据仓库的公开信息。

## 6. 正式写作阶段补充核验（2026-09-09）

用户指定的 Thirumalai (2020) 已包含在旧版39条文献中，因而直接核验并保留 `thirumalai2020methane`，没有添加重复条目；文献总数仍为57条。

完整阅读用户提供的 Thirumalai 11页原文，核对了其MMV统计量、甲烷与中国石笋差异、Barker synthetic record的讨论及解释边界。其他九条核对出版社/作者机构库的摘要及元数据；Dokken、Pedro另读本地发表版的相关页。具体可支持的论点及页码见 [literature_notes.md](literature_notes.md)。

| Citation key | 核验与修改 | 直接来源 |
|---|---|---|
| `thirumalai2020methane` | 作者、题名、GRL 47(9)、e2020GL087613、年份正确；规范 DOI 大小写。MMV并非本稿的条件事件率。 | [出版社](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020GL087613)、用户提供的原文 pp.1–11 |
| `sun2021persistent` | 10位作者、题名、卷页、DOI一致；主要论点为多记录的变率幅度调制。 | [出版社](https://www.nature.com/articles/s41561-021-00794-1) |
| `hodell20231` | 补齐最后一位作者 Eric W. Wolff，替代 `others`；其余一致。§3.1也统计事件数，不能说所有既有研究只分析振幅。 | [出版社](https://cp.copernicus.org/articles/19/607/2023/index.html) |
| `vettoretti2022atmospheric` | 化学式排版与期刊大小写修正；4位作者、卷页、DOI一致。 | [出版社](https://www.nature.com/articles/s41561-022-00920-7) |
| `zhang2021direct` | 8位作者、卷页、DOI一致；明确模型中具体描述的直接跃迁为暖转冷。 | [出版社](https://www.nature.com/articles/s41561-021-00846-6)、[作者机构库](https://orca.cardiff.ac.uk/id/eprint/145285/) |
| `pedro2022dansgaard` | 补齐最后一位作者 Kerim H. Nisancioglu，替代 `others`；11位作者、289卷、107599、DOI与本地发表版一致。 | [作者机构库](https://researchprofiles.ku.dk/en/publications/dansgaard-oeschger-and-heinrich-event-temperature-anomalies-in-th/)、本地原文 p.1 |
| `dokken2013dansgaard` | 5位作者、题名、28(3)、491–502、DOI与本地发表版一致。 | [出版社 DOI](https://doi.org/10.1002/palo.20042)、本地原文 pp.491–492 |
| `ditlevsen2005recurrence` | 3位作者、18(14)、2594–2603、年份与 DOI 一致；规范 DOI 大小写。 | [出版社](https://journals.ametsoc.org/abstract/journals/clim/18/14/jcli3437.1.xml) |
| `kleppin2015stochastic` | 5位作者、28(19)、7741–7763、年份与 DOI 一致；规范 DOI 大小写。模拟只有3次转折，应保留解释限度。 | [出版社](https://journals.ametsoc.org/view/journals/clim/28/19/jcli-d-14-00728.1.xml) |
| `lohmann2018random` | 题名的 `occurrence` 改为原文复数 `occurrences`；其余一致。作为条件率分析的近邻先行研究。 | [出版社](https://cp.copernicus.org/articles/14/609/2018/) |

这10条的本轮核验使用出版社或作者机构库及本地原文，没有将本轮未成功返回的 Crossref 请求当作证据。未修改旧项目中的 `.bib` 或任何原论文文件。

## 7. 首稿排版与论点复核

- 对正文和补充材料实际引用的37条文献，检查 `agu.bst` 转为句首大写时的表现，保护 Greenland、North Atlantic、Northern Hemisphere、Dansgaard–Oeschger、EPICA、SciPy、Python 等专名和化学式；规范继承条目中的 Science、Nature、Quaternary Science Reviews 期刊大小写。没有修改题名含义或 citation key。
- 将活动条目作者字段中连写的多字母缩写（如 `JB`、`ACM`）分开为 `J. B.`、`A. C. M.`，避免 BibTeX 将它们误作单个名字而漏掉后续首字母；没有推测或补造全名。
- 使用实际的 `agu.bst` 在临时目录 `/private/tmp/mcv_agu_bib_capitalization` 编译37条参考文献，BibTeX 无错误或警告，pdfLaTeX 连续编译通过。
- 阅读 Southon (2004) pp.1244、1246 与 Wolff et al. (2010) pp.2830–2831，核实 ss09/ss09sea 的早期 SPECMAP 年代约束以及 GICC05 模型延伸的继承关系。该证据支持有限的年代构建依赖，不能写成逐个 NGRIP 事件都经过岁差调谐；详见 [literature_notes.md](literature_notes.md)。
