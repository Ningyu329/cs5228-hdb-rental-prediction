# CS5228 HDB Rental Prediction

Private team repository for the CS5228-2610 Kaggle competition.

## Objective

Predict `MONTHLY_RENT` for Singapore HDB flats and evaluate models using RMSE.

The main approach is supervised regression:

`flat attributes + location + time + auxiliary information → monthly rent`

Clustering and other data-mining methods may support exploration or feature engineering, but they are not the primary prediction method.

## Project stages

### 1. Data audit and cleaning

- Inspect `train.csv` and `test.csv` for types, missing values, duplicates, anomalies, and train/test differences.
- Standardise text fields such as town, block, street, flat type, flat model, and furnishing status.
- Parse rental approval dates, normalise numerical columns, and derive basic fields such as flat age and fee indicators.
- Clean all auxiliary sources before use:
  - HDB block addresses, coordinates, and invalid completion years;
  - MRT station coordinates and station status;
  - school and mall coordinates;
  - COE currency values and monthly rounds;
  - stock-price dates and monthly aggregates.
- Preserve `data/raw/` unchanged and make each cleaning step reproducible by script.

### 2. Exploratory analysis and baseline

- Analyse rent distributions, time trends, town and flat-type differences, floor area, and flat age.
- Use a time-based validation split; retain a random split only as a secondary reference.
- Build simple baselines, followed by a supervised regression baseline such as CatBoost or LightGBM using main-table features.
- Record validation RMSE, model configuration, feature set, and create a correctly formatted test prediction file.

### 3. Midterm progress-report slides

- Present the problem definition, dataset summary, and data-cleaning decisions.
- Show key EDA findings, validation design, and baseline RMSE.
- Describe the auxiliary datasets and the planned location-feature approach.
- State current limitations, risks of data leakage, and next steps.

### 4. Address matching and location features

- Match `TOWN + BLOCK + STREET` to the HDB block data and report matching coverage.
- Add coordinates, planning area, region, completion year, and match-status indicators.
- Create distance and density features for open MRT stations, schools, and shopping malls.
- Optionally test geographic clustering as an additional regression feature.

### 5. Feature expansion and model improvement

- Add time, flat-age, fee, address-level, and location-derived features.
- Evaluate monthly COE features and carefully selected aggregated market indicators.
- Compare CatBoost, LightGBM/XGBoost, and an optional ensemble on the same validation split.
- Use feature importance and segment-level errors to guide further improvements.

### 6. Final evaluation and delivery

- Perform ablation experiments for main-table, location, facilities, and macro features.
- Analyse errors by town, flat type, age, size, time period, and address-match status.
- Train the chosen final approach, generate the Kaggle submission, and document results.
- Complete the final report with methodology, model comparison, findings, limitations, and reproducibility notes.

## 数据处理、EDA 与 Baseline

### 数据流

原始文件不能直接被后续流程覆盖；所有转换均由脚本完成，并按以下层级保存：

```text
data/raw/        → 原始 Kaggle 主数据和 auxiliary 数据，只读保存
data/interim/    → 格式统一、完成基础清洗的主表和辅助表
data/processed/  → 已按实验方案合并特征、可直接训练模型的数据表
```

典型输出包括 `train_clean`、`test_clean`、清洗后的 HDB/MRT/学校/商场/COE 表，以及不同特征版本的 `train_features`、`test_features`。这些数据文件不进入 Git，因为它们都必须能够由脚本从 `data/raw/` 重新生成。

### EDA 的作用

EDA（探索性数据分析）不是独立的预测方法，而是用来理解数据、发现质量问题并决定特征和验证策略。至少应检查：

- 租金的分布和极端值；
- 不同年月的租金趋势；
- 不同城镇、房型和房屋模型的租金差异；
- 建筑面积、房龄与租金的关系；
- 训练集和测试集在时间、区域和房型上的分布差异。

若租金存在明显时间趋势，应优先采用时间切分验证，而不是仅使用随机切分。

### Baseline 与特征逐步接入

Baseline 的目标是先建立一个可靠的参照。它使用基础清洗后的主数据训练监督回归模型，并以 `MONTHLY_RENT` 为目标；首版不必接入 auxiliary 数据。CatBoost 或 LightGBM 是合适的首个模型选择。

后续在完全相同的验证集上逐步比较特征版本：

1. **Baseline**：清洗后的主表字段；
2. **时间与房屋特征**：租赁年月、房龄、面积、费用等；
3. **地理特征**：HDB 坐标、区域、到 MRT 的距离；
4. **设施特征**：学校和商场距离或密度；
5. **宏观特征**：COE 与经过聚合筛选的市场指标；
6. **最终模型**：只保留能稳定改善验证 RMSE 的特征，并可测试模型集成。

每个版本都应记录特征集合、模型参数和验证 RMSE。聚类可作为区域或社区类型的额外特征，但不能替代带有租金标签训练的监督回归模型。

`postprocessing` 指模型预测后的结果处理，例如将不合理的负租金截断；它不用于存放清洗后的数据。

## Repository layout

- `src/data/`: cleaning and data-integration scripts
- `src/features/`: feature-engineering scripts
- `src/models/`: training, evaluation, and prediction scripts
- `notebooks/`: exploratory analysis only
- `configs/`: shared model and pipeline settings
- `reports/`: figures, progress-report slides, and written reports

Raw Kaggle competition data, local models, and submission files are excluded from Git. Each authorised team member should obtain competition data through Kaggle.

## 文件结构说明

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
├── notebooks/               # EDA 和展示用 Notebook；正式流程应迁移到 src/
├── configs/                 # 模型参数、数据路径和实验配置
├── reports/
│   ├── figures/             # 报告与 slides 使用的图表
│   └── progress_report/     # 中期报告 slides 和相关材料
├── submissions/             # 本地生成的 Kaggle 提交文件；不上传 Git
├── README.md                # 项目目标、流程和协作说明
└── .gitignore               # 排除数据、模型、提交文件和凭据
```

原则：原始数据只保存在 `data/raw/`；所有清洗、特征工程和训练结果都应能由 `src/` 中的脚本复现。Notebook 用于探索和呈现，不应是唯一的正式实现；密钥、Kaggle 凭据和私有竞赛数据不得提交到仓库。

## Collaboration workflow

1. Create a focused branch from `main`.
2. Keep changes reproducible and run relevant checks locally.
3. Open a pull request for review before merging into `main`.
4. Record every experiment's data version, validation strategy, feature set, model configuration, and RMSE.
