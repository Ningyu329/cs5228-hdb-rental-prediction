# CS5228 HDB 组屋月租预测

本仓库用于 CS5228-2610 Kaggle 私有竞赛的小组协作。

## 项目目标

预测新加坡 HDB 组屋的月租金 `MONTHLY_RENT`，并以均方根误差（RMSE）评估模型表现。

项目以监督学习回归为主线：

`房屋属性 + 地理位置 + 时间 + 辅助信息 → 月租金`

聚类和其他数据挖掘方法可用于探索或构造特征，但不是主要预测方法。

## 第 4、5 阶段交付索引

本次交付包括地址与位置特征、固定 Ridge 消融实验、阶段 5 模型/辅助特征比较，
以及后续 CatBoost / LightGBM 参数调优。保留各轮独立目录，不用新分数覆盖历史结果。

| 内容 | 代码入口 | 关键结果 |
| --- | --- | --- |
| 第 4 阶段：地址匹配与位置特征 | [build_location_features.py](src/features/build_location_features.py) | [覆盖率与限制](reports/location_features/location_features_report.md)、[配置与输入指纹](reports/location_features/location_features_manifest.json) |
| 位置特征 Ridge 消融 | [run_location_ablation.py](src/models/run_location_ablation.py) | [报告](reports/experiments/location_ablation_v1/location_ablation_report.md)、[24 组结果](reports/experiments/location_ablation_v1/ablation_results.csv) |
| 第 5 阶段：非线性模型、特征扩展 | [build_stage5_features.py](src/features/build_stage5_features.py)、[run_stage5.py](src/models/run_stage5.py) | [报告与图表](reports/experiments/stage5_v1/stage5_report.md)、[46 条结果](reports/experiments/stage5_v1/results.csv) |
| CatBoost / LightGBM 调优 | [run_tree_tuning.py](src/models/run_tree_tuning.py) | [调优报告](reports/experiments/tree_tuning_v1/tuning_report.md)、[前后对比](reports/experiments/tree_tuning_v1/comparison.csv) |
| 输出核验与测试 | [inspect_stage5.py](src/models/inspect_stage5.py)、[inspect_tree_tuning.py](src/models/inspect_tree_tuning.py)、[tests](tests) | [阶段 5 核验](reports/experiments/stage5_v1/output_validation.json)、[调优核验](reports/experiments/tree_tuning_v1/output_validation.json) |

关键结论（主时间验证为 2025-01 至 2025-03，RMSE 单位为 SGD/月）：

- 训练与测试地址匹配率及有效坐标覆盖率均为 100%；这不代表辅助信息满足历史可用性。
- 位置特征对固定 Ridge 的改善较小；含 MRT 的最低 RMSE 为 562.0281，因使用静态开放状态，仅作探索性结果。
- 阶段 5 通过较早季度选择的 housing 50/50 融合 RMSE 为 452.1702；不按主验证最低分事后换选。
- 本轮调优后，较早季度选定的新 50/50 融合 RMSE 为 **450.4735**；相对原融合改善约 0.38%，MAE 基本不变。
- 2025 年第一季度已在多个阶段被查看，不是完全未触碰的最终测试；尚未证明对测试集远期月份同样有效。

**Git 中包含什么：**源代码、44 项合成数据测试、依赖版本、Markdown 报告、汇总 CSV、
配置/选择记录、核验摘要，以及 3 张阶段 5 结果图。`.gitignore` 只逐项放行经过检查的
汇总 CSV，仍排除训练/测试/辅助数据、加工后的数据、逐行真实租金与预测、月度宏观特征表、
训练模型、提交文件和本地构建依赖。仅克隆仓库无法直接重算历史核验：逐行预测与模型必须先在本地重建。

### 从私有数据复现

建议使用 Python 3.12 独立环境，安装 `requirements-stage5.txt`；图表依赖 matplotlib。
私有数据目录应包含 `train.csv`、`test.csv` 与 `auxiliary/`，不要提交到 Git。
以下命令在仓库根目录运行，输入路径替换为自己的数据位置：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-stage5.txt
.\.venv\Scripts\python.exe src/data/clean_data.py --input-dir "D:\path\to\competition-data"
.\.venv\Scripts\python.exe src/features/build_location_features.py
.\.venv\Scripts\python.exe src/models/run_location_ablation.py
.\.venv\Scripts\python.exe src/models/run_stage5.py
.\.venv\Scripts\python.exe src/models/inspect_stage5.py
.\.venv\Scripts\python.exe src/models/run_tree_tuning.py --output-dir reports/experiments/tree_tuning_local
.\.venv\Scripts\python.exe src/models/inspect_tree_tuning.py --output-dir reports/experiments/tree_tuning_local
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

调优依赖 `stage5_v1` 的逐行基准预测，必须先运行阶段 5，不能只下载汇总结果跳过训练。
前面的默认命令会在工作区重新生成阶段 4/5 报告及对应配置；Git 保留已提交的历史版本。
调优使用新的 `tree_tuning_local` 目录，避免与本次发布的历史调优目录混用。
输入、版本、源码字节或配置不一致时，历史指纹不应被强行改写为“核验通过”；应重新运行对应实验。
`.gitattributes` 固定被核验源码及结果的换行策略，以降低跨平台检出造成的无意义指纹差异。

## 项目阶段

### 1. 数据审计与清洗

- 检查 `train.csv` 与 `test.csv` 的数据类型、缺失值、重复记录、异常值及分布差异。
- 统一城镇、楼号、街道、房型、房屋模型和装修状态等文本字段。
- 解析租赁批准日期、规范数值字段，并构造房龄等基础特征；检查费用字段是否存在足够变异，仅在有信息量时构造费用标记。
- 清洗所有辅助数据：
  - HDB 楼栋地址、坐标及无效竣工年份；
  - MRT 站点坐标与开放状态；
  - 学校与商场坐标；
  - COE 货币字段与月度轮次；
  - 股票价格的日期与数值字段；月度聚合指标留待阶段 5 按验证结果构造。
- 保持 `data/raw/` 不变，并使每一步清洗都能由脚本重复执行。

### 2. 探索性数据分析与基线模型

- 分析租金分布、时间趋势、城镇和房型差异、建筑面积及房龄。
- 以时间切分作为主要验证方案；随机切分仅作辅助参考。
- 先建立简单基线，再用主表特征训练 CatBoost 或 LightGBM 等监督回归基线模型。
- 记录验证 RMSE、模型配置和特征集合，并生成格式正确的测试集预测文件。

当前阶段可运行：

```bash
python src/eda/run_eda.py
python src/models/train_baseline.py
```

EDA 输出在 `reports/figures/` 和 `reports/tables/`；baseline 的 RMSE 与配置在
`reports/experiments/`，测试集预测在 `submissions/`。提交文件使用竞赛要求的
`Id,Predicted` 两列，其中 `Id` 按 `test.csv` 原始行顺序从 0 编号。首要验证集是训练数据的最后
3 个批准月份，随机 80/20 切分只作为辅助参考。`train_baseline.py` 在安装
CatBoost 时优先使用 CatBoost；若当前环境没有 CatBoost，则自动运行可复现的
one-hot Ridge fallback，并在结果中标明实际模型。

### 3. 中期进展汇报幻灯片

- 说明问题定义、数据概况和数据清洗决策。
- 展示关键探索性分析结论、验证设计和基线 RMSE。
- 介绍辅助数据及规划中的位置特征方案。
- 说明当前局限、数据泄漏风险和下一步计划。

### 4. 地址匹配与位置特征

- 用 `TOWN + BLOCK + STREET` 与 HDB 楼栋数据匹配，并报告匹配覆盖率。
- 加入坐标、规划区、区域、竣工年份和地址匹配状态。
- 构造到已开放 MRT 站、学校和商场的距离与密度特征。
- 可选：测试地理聚类，并将聚类结果作为额外回归特征。

已实现 `src/features/build_location_features.py`。先将竞赛数据放回 `data/raw/`
并运行清洗，再生成第一版位置特征（需要 NumPy、pandas，与已有 EDA 一致）：

```bash
python src/data/clean_data.py
python src/features/build_location_features.py
```

输入为 `data/interim/train_clean.csv`、`test_clean.csv`，以及 `auxiliary/` 下清洗后的
HDB、MRT、学校、商场表。输出包括：

- `data/processed/train_location_features.csv` 和 `test_location_features.csv`：保留原行序、
  重复观测和所有原始主表字段，追加位置特征，不修改原始数据。
- `data/processed/train_unmatched_addresses.csv` 和 `test_unmatched_addresses.csv`：未匹配地址，
  便于检查，仍按私有数据排除在 Git 之外。
- `reports/location_features/location_features_report.md`：本次训练/测试行级匹配率、
  按城镇覆盖情况、有效坐标数量及数据限制。
- `reports/location_features/location_features_manifest.json`：新增特征列表、输入 SHA-256、
  半径设置及辅助数据去重/无效坐标统计。

新增特征包括 `LATITUDE`、`LONGITUDE`、`SUBZONE`、`PLANNING_AREA`、`REGION`、
`YEAR_COMPLETED`、`MAX_FLOOR`、`POSTAL_CODE`、`ADDRESS_MATCHED`、
`LOCATION_COORDINATES_VALID`，以及 `BUILDING_AGE` 和相关缺失标记。
`BUILDING_AGE` 从竣工年计算，与从租约起始年计算的 `FLAT_AGE` 不同。
距离特征为 `NEAREST_OPEN_MRT_DISTANCE_KM`、`NEAREST_SCHOOL_DISTANCE_KM`、
`NEAREST_MALL_DISTANCE_KM`；密度代理为各类设施在 **500m / 1000m** 内的数量，
例如 `OPEN_MRT_COUNT_WITHIN_1000M`。距离使用 Haversine 大圆距离，单位 km，
不是步行路程。MRT 同坐标的多个线路代码仅计一个位置。

默认 `--mrt-mode snapshot` 只使用当前 `STATUS=open` 的站点。这排除了规划站点，
但**不能保证消除历史时间泄漏**：当前已开放的站点在 2021 年可能还未开放。
正式时间验证建议准备以站点代码为键的 `CODE,OPENING_DATE` CSV（ISO 日期，例如
`2020-01-31`），然后运行：

```bash
python src/features/build_location_features.py --mrt-mode historical --mrt-opening-dates path/to/mrt_opening_dates.csv
```

也支持在清洗后的 MRT 输入表直接附带 `OPENING_DATE`。historical 模式要求所有 open
站点都有有效开放日期，仅使用在**租赁批准月份第一天**已开放的站点，同月中途开放的站点
从下月起计入。学校、商场和 HDB 区域属性仍来自静态快照，报告会明确这一限制。
地址不匹配或坐标无效时，保留该行并将距离与计数置空，不以 0 冒充已知距离。
HDB 重复地址键会报错，避免合并导致行数膨胀。尚未实现可选地理聚类。

可通过 `--interim-dir`、`--output-dir`、`--report-dir` 指定其他目录，
通过 `--radii-m 500 1000 2000` 调整半径。没有输入 CSV 时脚本会明确列出缺失文件，
不会生成虚构的覆盖率或 RMSE。测试使用小型合成数据：

```bash
python -m unittest discover -s tests -v
```

**建模衔接：**原 `train_baseline.py` 仍只读取 `data/interim/` 的主表字段，
重跑它不会自动使用这些位置特征。阶段 5 应读取本阶段的 processed 文件、显式选择新增特征，
仅在训练折拟合填补器与编码器，并在同一时间验证集上对比主表 baseline 和位置特征版本。
本阶段不宣称已优于原 Ridge temporal RMSE 562.59。

### 5. 特征扩展与模型优化

- 加入时间、房龄、费用、地址层级和位置衍生特征。
- 评估月度 COE 特征及经筛选聚合后的市场指标。
- 在同一验证集上比较 CatBoost、LightGBM/XGBoost 和可选的模型集成。
- 通过特征重要性与分组误差分析指导优化。

已实现位置特征的 Ridge 消融实验脚本，固定原基线参数，比较以下六组特征：
主表、主表加 HDB 属性、加 MRT、加 MRT 和学校、加全部设施、加学校与商场但不加 MRT。
`ADDRESS_KEY` 和 `POSTAL_CODE` 不作为新增输入。运行：

```bash
python src/models/run_location_ablation.py --reference-rmse 562.5887838225482
```

默认使用最新 3 个月作为主验证集，额外回测之前两个互不重叠的 3 个月窗口，并报告
随机 80/20 切分作为辅助。每个时间窗口均只使用其之前的记录训练，填补器、标准化器和
One-hot 编码器也仅在各训练折拟合。共运行 6 组特征乘以 4 个划分的 24 组实验。
可用 `--backtest-windows`、`--skip-random` 和 `--variants` 控制范围，但必须保留 `main_only`。

结果位于 `reports/experiments/location_ablation_v1/`，包括：

- `location_ablation_report.md`：各组 RMSE、同环境基线改善、回测与时间可用性限制。
- `ablation_results.csv`：RMSE、MAE、预测偏差、运行耗时及各划分的日期与行数。
- `experiment_manifest.json`：特征集合、配置、依赖版本、输入文件 SHA-256。
- `group_errors.csv`：主表及主要候选按城镇、房型、月份和区域的误差。
- `primary_validation_predictions.csv`：主验证逐行预测，包含原训练行号与真实租金，属于
  私有 CSV，受 `.gitignore` 保护。

脚本不覆盖原 baseline 配置或生成竞赛提交。先前 RMSE 仅作历史参考，改善以本次同环境
复跑为准。MRT snapshot 方案分数仅作探索性比较，不能据此宣称排除了历史时间泄漏；
报告另列不含 MRT 的最低分候选。

第 5 阶段完整实验入口为 `src/models/run_stage5.py`，依赖固定在
`requirements-stage5.txt`（本次运行使用 Python 3.12 的项目独立环境）。Windows 运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-stage5.txt
.\.venv\Scripts\python.exe src/models/run_stage5.py
```

该实验包含：

- 原始 Ridge 和限制时间外推的样条 Ridge，用于检查线性时间趋势是否导致整体高估。
- CatBoost 与 LightGBM 比较主表、房屋/地址/HDB 属性及学校/商场，宏观扩展与
  单独标注的 MRT 快照方案由 LightGBM 完整筛选，以控制训练成本。
  房屋扩展包括对数面积、月份周期、每房间面积、固定房龄分组和
  `TOWN + BLOCK + STREET` 组合地址。FEE/FURNISHED 无变异，不增加无信息标记。
- 月度 COE 各轮溢价均值与投标/配额比，及股票等权月收益率、平均日波动率、上涨证券占比
  和证券覆盖数。股票使用未复权 CLOSE，不用未来价格填补缺失。
- 宏观特征滞后 **2 个月**。预测完整验证窗口时冻结于训练截止时可用的源月份，
  不能直接使用辅助表中已经包含的未来月份数值。由于缺乏真实发布时间，此规则仍是可用性假设。
- 每个外层训练集内部保留最后 3 个月进行早停轮数选择，再重训完整外层训练数据。
  CatBoost/LightGBM 等权融合固定为 50/50，不在主验证集调权重。
- 仅用 2024 年第三、四季度两个较早验证窗口的合并 RMSE 选择候选，在计算 2025 年第一季度
  主验证分数前写入 `selection_locked.json`。MRT 快照方案不参与推荐选择，随机切分仅为辅助。

输出位于 `reports/experiments/stage5_v1/`：`stage5_report.md`、`results.csv`、
`experiment_manifest.json`、`selection_backtests.csv`、逐窗口预测、分组误差及特征重要性。
特征重要性包括树模型训练重要性和主验证固定样本的成组置换重要性，不把重要性解释为因果效应。
`models/stage5/` 保存主验证截止时训练的候选诊断模型和输入 schema，受 Git 忽略规则保护，
不是阶段 6 的全量训练最终模型。本脚本不生成或上传竞赛提交。

可通过 `--tree-iterations` 限制早停搜索上限，`--threads` 限制 CPU 并行度，
`--workers` 控制同时训练的独立候选数（默认 3，每个候选 4 个线程），
`--schemes` 缩小特征比较范围，或 `--output-dir` 将不同实验配置分开保存。
已完成拟合缓存于 `models/stage5_cache/`，按数据、配置、依赖版本和训练/特征实现文件的 SHA-256
生成键，中断后同配置重跑可恢复，修改实现不会误用旧缓存。
实验完成后独立核对导出的预测/误差，并生成模型对比、特征组重要性与月度偏差图片：

```powershell
.\.venv\Scripts\python.exe src/models/inspect_stage5.py
```

#### CatBoost / LightGBM 参数调优

保持阶段 5 的 housing 特征不变，执行有界时间验证搜索：

```powershell
.\.venv\Scripts\python.exe src/models/run_tree_tuning.py
.\.venv\Scripts\python.exe src/models/inspect_tree_tuning.py
```

先在 2024 年第二季度筛选 8 组配置，包括原参数对照、CatBoost 更高轮数/深度，
以及 LightGBM 不同叶子数、学习率、叶子最小样本数与正则强度。
每种算法选出一个新配置，在第三、四季度复核，并与原算法、预设的 25/50/75%
融合权重共同比较；按两个较早窗口合并 RMSE 锁定选择后，才评估 2025 年第一季度。
每次外层训练仍使用内部最后三个月选择早停轮数，不用外层验证标签早停。
旧的 50/50 融合始终可以被保留；调参不保证改善，也不据主验证结果事后换选。

结果单独保存在 `reports/experiments/tree_tuning_v1/`，包含筛选表、逐窗口预测、
锁定选择、前后比较、分组误差和 `tuning_report.md`，不覆盖阶段 5。
原模型预测仅在输入数据与实现校验一致、验证行序及标签完全对齐时复用。
诊断模型缓存于 `models/tree_tuning_cache/`，同配置中断后重跑可以恢复；
更改配置或实现时使用新的 `--output-dir`，避免覆盖已有实验。
默认同时运行 3 个候选、每个 4 线程，可通过 `--workers` 和 `--threads` 降低资源占用。
核查脚本独立重算误差、核对选择规则，并验证保存模型重新加载后的逐行预测一致。
本轮不读取测试数据、不训练全量最终模型、不生成或上传竞赛提交。

本轮实际主时间验证（2025-01 至 2025-03）结果：CatBoost 从 457.2788 降至
455.2215，LightGBM 从 455.1987 降至 453.0133；较早窗口预先选出的新 50/50
融合从原来的 452.1702 降至 **450.4735**，改善约 0.38%。融合 MAE 基本不变
（331.1601 → 331.1260），不能将这一结果解释为所有误差指标均明显改善。
新 CatBoost 主验证训练使用 depth=8、learning_rate=0.05、l2_leaf_reg=10、800 轮，
仍达到本次上限；新 LightGBM 使用 num_leaves=15、min_child_samples=80、
reg_lambda=10、learning_rate=0.04，内部早停选择 481 轮。
44 项单元测试通过；独立核查 38 条外层结果及 326,504 个导出预测值，
两个模型重新加载后的主验证逐行预测一致。完整结果与限制见
`reports/experiments/tree_tuning_v1/tuning_report.md`。

### 6. 最终评估与交付

- 对主表、位置、设施和宏观特征进行消融实验。
- 按城镇、房型、房龄、面积、时间和地址匹配状态分析误差。
- 训练最终模型，生成 Kaggle 提交文件并记录结果。
- 完成包含方法、模型比较、发现、局限与复现说明的最终报告。

## 数据处理、探索性分析与基线模型

### 数据流

原始文件不能被后续流程覆盖；所有转换均由脚本完成，并按以下层级保存：

```text
data/raw/        → 原始 Kaggle 主数据和辅助数据，只读保存
data/interim/    → 格式统一、完成基础清洗的主表和辅助表
data/processed/  → 已按实验方案合并特征、可直接训练模型的数据表
```

典型输出包括 `train_clean`、`test_clean`、清洗后的 HDB/MRT/学校/商场/COE 表，以及不同特征版本的 `train_features`、`test_features`。这些数据文件不进入 Git，因为它们都必须能够由脚本从 `data/raw/` 重新生成。

### 探索性数据分析的作用

探索性数据分析不是独立的预测方法，而是用来理解数据、发现质量问题并决定特征和验证策略。至少应检查：

- 租金的分布和极端值；
- 不同年月的租金趋势；
- 不同城镇、房型和房屋模型的租金差异；
- 建筑面积、房龄与租金的关系；
- 训练集和测试集在时间、区域和房型上的分布差异。

由于测试集包含训练集之后的批准月份，应优先采用按批准月份切分的验证方案。这是
为了模拟未来月份的分布变化，并不意味着把数据当成纯时间序列；每条记录仍是一个
带有时间属性的横截面租赁观测，且没有可识别的具体 flat ID。

### 基线模型与特征逐步接入

基线模型的目标是先建立一个可靠的参照。它使用基础清洗后的主数据训练监督回归模型，并以 `MONTHLY_RENT` 为目标；首版不必接入辅助数据。CatBoost 或 LightGBM 是合适的首个模型选择。

后续在完全相同的验证集上逐步比较以下特征版本：

1. **基线模型**：清洗后的主表字段；
2. **时间与房屋特征**：租赁年月、房龄、面积、费用等；
3. **地理特征**：HDB 坐标、区域、到 MRT 的距离；
4. **设施特征**：学校和商场距离或密度；
5. **宏观特征**：COE 与经过聚合筛选的市场指标；
6. **最终模型**：只保留能稳定改善验证 RMSE 的特征，并可测试模型集成。

每个版本都应记录特征集合、模型参数和验证 RMSE。聚类可作为区域或社区类型的额外特征，但不能替代带有租金标签训练的监督回归模型。

预测后处理指模型输出后的结果处理，例如将不合理的负租金截断；它不用于存放清洗后的数据。

## 仓库结构

```text
project/
├── data/
│   ├── raw/                 # Kaggle 原始数据及辅助数据；只读，不修改，不上传 Git
│   ├── interim/             # 清洗、地址匹配等中间结果；可由脚本重新生成，不上传 Git
│   └── processed/           # 最终用于建模的特征数据；可由脚本重新生成，不上传 Git
├── src/
│   ├── data/                # 数据审计、清洗、合并辅助数据的脚本
│   ├── features/            # 日期、地址、地理距离、COE 等特征工程脚本
│   ├── models/              # 训练、验证、预测和生成提交文件的脚本
│   └── utils/               # 可复用的路径、评估、日志等辅助函数
├── notebooks/               # 探索性分析和展示用 Notebook；正式流程应迁移到 src/
├── configs/                 # 模型参数、数据路径和实验配置
├── reports/
│   ├── figures/             # 报告与幻灯片使用的图表
│   └── progress_report/     # 中期报告幻灯片和相关材料
├── submissions/             # 本地生成的 Kaggle 提交文件；不上传 Git
├── README.md                # 项目目标、流程和协作说明
└── .gitignore               # 排除数据、模型、提交文件和凭据
```

原则：原始数据只保存在 `data/raw/`；所有清洗、特征工程和训练结果都应能由 `src/` 中的脚本复现。Notebook 用于探索和呈现，不应是唯一的正式实现；密钥、Kaggle 凭据和私有竞赛数据不得提交到仓库。

## 协作流程

1. 从 `main` 创建聚焦于单一任务的分支。
2. 保证改动可复现，并在本地完成相关检查。
3. 合并到 `main` 前创建拉取请求，供组员审阅。
4. 记录每次实验的数据版本、验证方案、特征集合、模型配置和 RMSE。
