# 第 5 阶段：特征扩展与模型优化

## 验证与选择规则

- 使用 2024-07 至 2024-09、2024-10 至 2024-12 两个较早窗口的合并 RMSE 选择候选。
- 2025-01 至 2025-03 为主验证窗口，不用于选择模型、早停轮数、融合权重或特征。先前实验已查看过该窗口，因此它不是完全未触碰的最终测试集。
- CatBoost、LightGBM 仅在各外层训练集内部保留最后 3 个月用于早停，再按选定轮数重训全部外层训练数据。
- 融合固定为相同特征 CatBoost/LightGBM 各 50%，不根据主验证调权重。
- 所有宏观特征滞后两个月，外层和内层验证均按各自训练截止时间冻结可用源月份。未来验证月份的 COE 和股票值不参与预测。

## 主时间验证结果

| 候选 | RMSE | MAE | 平均预测偏差 | 较早窗口选中 |
| --- | ---: | ---: | ---: | --- |
| blend_main | 451.6188 | 329.2781 | -23.72 |  |
| blend_housing | 452.1702 | 331.1601 | -30.59 | 是 |
| blend_amenities | 452.5783 | 330.8470 | -28.19 |  |
| lightgbm_housing | 455.1987 | 332.9281 | -26.55 |  |
| lightgbm_main | 455.5022 | 331.9834 | -19.66 |  |
| lightgbm_amenities | 455.9756 | 332.5209 | -20.01 |  |
| lightgbm_coe | 456.5514 | 332.9769 | -16.86 |  |
| lightgbm_market | 456.6153 | 332.6809 | -18.69 |  |
| catboost_amenities | 456.9819 | 335.6525 | -36.38 |  |
| catboost_housing | 457.2788 | 336.0324 | -34.63 |  |
| lightgbm_mrt_snapshot | 457.4575 | 333.1983 | -18.97 |  |
| catboost_main | 458.3400 | 335.7604 | -27.79 |  |
| ridge_spline_main | 465.2389 | 335.5900 | +4.65 |  |
| ridge_main | 562.6666 | 406.7216 | +316.44 |  |

## 结果解读

- 较早窗口选出的候选是 blend_housing。主验证 RMSE 为 452.1702，同环境 Ridge 为 562.6666，改善 19.64%。
- 平均预测偏差从 Ridge 的 +316.44 变为候选的 -30.59。正值表示高估月租。
- 模型/特征选择详见 selection_backtests.csv。主验证最低分不一定是按较早窗口选出的候选，保留这一差异，避免事后换选。

## 主要特征组

| 特征组 | 打乱后样本 RMSE 增加 |
| --- | ---: |
| housing | 243.8911 |
| hdb | 69.1611 |
| address | 31.4140 |
| time | 0.4464 |

重要性来自主验证固定样本的成组置换，重复 2 次。只用于解释，不再用其选择特征。相关特征可能互相替代，不能解释为因果效应。

## 时间、宏观与数据限制

- mrt_snapshot 方案只作探索，不能参与推荐选择，因为缺少历史开放日期。学校、商场、部分楼栋属性也来自静态快照。
- 月度 COE 使用各轮溢价均值及总投标/总配额比。股票使用未复权 CLOSE 的等权月收益率、平均日波动率、上涨证券占比和证券覆盖数，不填补未来价格。
- 宏观表没有真实发布时间。两个月滞后是保守可用性假设，不构成无泄漏保证。未复权价格的拆股等公司行动也可能造成跳变。
- 冻结后的宏观值在一个验证窗口内可能恒定，树模型的未来时间字段也可能落在相同叶子。局部置换重要性接近 0 不代表这些变量在历史训练中没有作用。
- FEE/FURNISHED 在当前数据中没有变异，因此不增加费用或装修标记。
- 季度验证不能保证延伸到 2026-07 的长预测跨度表现。需要在阶段 6 做更长窗口评估及最终提交。
- 本轮只在预设参数下优化训练轮数。若早停轮数达到搜索上限，需另做更高上限的稳健性检查，不能宣称参数已达到全局最优。
- 尚未读取测试标签，也未生成/上传新的竞赛提交。保存的模型是主验证训练截止时的诊断模型，不是全量训练最终模型。

## 输出

results.csv 保存全部外层实验；group_errors.csv 保存按城镇、房型、模型、面积、房龄、月份、区域和地址匹配状态的误差。
group_permutation_importance.csv 为候选特征组重要性。native_feature_importance.csv 为树模型训练重要性，CatBoost 为 PredictionValuesChange，LightGBM 为 gain，不能直接跨算法比较原始数值。

模型 API 依据 [CatBoost 官方文档](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor) 与 [LightGBM 官方文档](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html)。

## 候选模型的分组误差诊断

下表仅为主验证描述，不据此重新挑选模型。城镇按至少 100 条记录筛选后列出误差最高的 4 组，避免按极小样本排序。

| 城镇 | 验证记录数 | RMSE | 平均预测偏差 |
| --- | ---: | ---: | ---: |
| jurong west | 620 | 548.76 | +22.40 |
| kallang/whampoa | 342 | 541.49 | -3.83 |
| bukit merah | 530 | 508.53 | -62.77 |
| toa payoh | 341 | 498.81 | -55.47 |

| 房型 | 验证记录数 | RMSE | 平均预测偏差 |
| --- | ---: | ---: | ---: |
| 1-room | 4 | 204.14 | -39.69 |
| 2-room | 242 | 408.90 | -15.64 |
| 3-room | 2915 | 425.90 | -30.80 |
| 4-room | 3201 | 451.54 | -28.75 |
| 5-room | 2100 | 471.28 | -23.72 |
| executive | 462 | 542.77 | -80.90 |

后续应检查高误差城镇与大户型的残差分布，并在较早时间窗口验证任何修正方案。稀有房型样本少，不宜仅按主验证分数判断模型优劣。

## 选定组件的训练重要性

| 组件 | 特征 | 重要性数值 | 方法 |
| --- | --- | ---: | --- |
| catboost_housing | RENT_MONTH_INDEX | 40.31 | PredictionValuesChange |
| catboost_housing | FLAT_TYPE | 25.57 | PredictionValuesChange |
| catboost_housing | LATITUDE | 5.533 | PredictionValuesChange |
| catboost_housing | RENT_YEAR | 3.653 | PredictionValuesChange |
| catboost_housing | SUBZONE | 2.806 | PredictionValuesChange |
| lightgbm_housing | RENT_MONTH_INDEX | 2.874e+11 | gain |
| lightgbm_housing | FLAT_TYPE_ROOMS | 1.186e+11 | gain |
| lightgbm_housing | SUBZONE | 5.548e+10 | gain |
| lightgbm_housing | RENT_YEAR | 3.677e+10 | gain |
| lightgbm_housing | STREET | 3.609e+10 | gain |

训练期的时间字段重要性较高，而主验证局部置换时间组的作用较小，两者衡量范围不同，不能将局部重要性解释为时间趋势不重要。

## 结果图片

![01_model_comparison](figures/01_model_comparison.png)

![02_group_importance](figures/02_group_importance.png)

![03_monthly_bias](figures/03_monthly_bias.png)
