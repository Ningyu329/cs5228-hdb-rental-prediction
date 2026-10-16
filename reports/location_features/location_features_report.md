# 地址匹配与位置特征运行报告

本报告仅依据本次运行的输入生成，不使用历史审计匹配率代替本次结果。

| 数据集 | 行数 | 匹配行数 | 地址匹配率 | 有效坐标行数 | 缺失建筑年龄行数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 150,000 | 150,000 | 100.00% | 150,000 | 159 |
| test | 50,000 | 50,000 | 100.00% | 50,000 | 0 |

## 特征与处理规则

- 地址按 TOWN + BLOCK + STREET 标准化后精确匹配，缺失地址不匹配，不做模糊猜测。
- HDB 地址键必须唯一。保留主表行数、行序、重复观测和训练标签，不使用租金计算位置特征。
- 保留坐标、规划区、区域、竣工年份、最高楼层及邮编。ADDRESS_MATCHED 与 LOCATION_COORDINATES_VALID 分开标记。
- BUILDING_AGE = 租赁批准年份 - YEAR_COMPLETED，与租约起始年计算的 FLAT_AGE 含义不同。无效、缺失或负建筑年龄保留为空并添加缺失标记。
- 距离使用 Haversine 大圆距离，单位 km。半径内设施数量作为密度代理，包含半径边界，不代表步行距离或真实服务质量。
- MRT 相同坐标仅计一次，避免换乘站多个线路代码重复计数；学校、商场按名称和坐标去重。
- 无效或未匹配坐标的距离与计数保留为空。无可用设施时最近距离为空，有效坐标的设施计数为 0。

## 时间可用性与限制

- MRT snapshot 模式仅筛选当前 STATUS=open，不能证明站点在历史租赁月份已开放，仍有时间泄漏风险。正式时间验证应提供开放日期并使用 historical 模式。
- 学校、商场和 HDB 区域属性来自静态辅助表，没有完整历史版本。不能宣称整套位置特征已完全消除时间泄漏。
- 特征生成不拟合填补器、编码器或聚类。后续建模必须仅在训练折拟合这些转换。

## 辅助数据检查

| 来源 | 输入行数 | 有效唯一位置数 | 无效坐标行数 |
| --- | ---: | ---: | ---: |
| mrt | 243 | 192 | 0 |
| school | 337 | 337 | 0 |
| mall | 113 | 113 | 0 |

## 按城镇的地址匹配

| 数据集 | 城镇 | 行数 | 匹配行数 |
| --- | --- | ---: | ---: |
| train | ang mo kio | 8283 | 8283 |
| train | bedok | 8907 | 8907 |
| train | bishan | 3306 | 3306 |
| train | bukit batok | 5605 | 5605 |
| train | bukit merah | 8270 | 8270 |
| train | bukit panjang | 3717 | 3717 |
| train | bukit timah | 438 | 438 |
| train | central | 2278 | 2278 |
| train | choa chu kang | 5300 | 5300 |
| train | clementi | 5429 | 5429 |
| train | geylang | 4504 | 4504 |
| train | hougang | 7115 | 7115 |
| train | jurong east | 4024 | 4024 |
| train | jurong west | 10482 | 10482 |
| train | kallang/whampoa | 5689 | 5689 |
| train | marine parade | 1637 | 1637 |
| train | pasir ris | 4089 | 4089 |
| train | punggol | 6668 | 6668 |
| train | queenstown | 6115 | 6115 |
| train | sembawang | 3553 | 3553 |
| train | sengkang | 9773 | 9773 |
| train | serangoon | 3398 | 3398 |
| train | tampines | 9942 | 9942 |
| train | toa payoh | 5988 | 5988 |
| train | woodlands | 7577 | 7577 |
| train | yishun | 7913 | 7913 |
| test | ang mo kio | 2664 | 2664 |
| test | bedok | 2803 | 2803 |
| test | bishan | 982 | 982 |
| test | bukit batok | 2057 | 2057 |
| test | bukit merah | 2832 | 2832 |
| test | bukit panjang | 1228 | 1228 |
| test | bukit timah | 156 | 156 |
| test | central | 831 | 831 |
| test | choa chu kang | 1687 | 1687 |
| test | clementi | 1883 | 1883 |
| test | geylang | 1445 | 1445 |
| test | hougang | 2298 | 2298 |
| test | jurong east | 1371 | 1371 |
| test | jurong west | 3663 | 3663 |
| test | kallang/whampoa | 1916 | 1916 |
| test | marine parade | 493 | 493 |
| test | pasir ris | 1329 | 1329 |
| test | punggol | 2286 | 2286 |
| test | queenstown | 2148 | 2148 |
| test | sembawang | 1298 | 1298 |
| test | sengkang | 3231 | 3231 |
| test | serangoon | 1099 | 1099 |
| test | tampines | 3198 | 3198 |
| test | toa payoh | 2055 | 2055 |
| test | woodlands | 2385 | 2385 |
| test | yishun | 2662 | 2662 |

## 后续验证

使用相同时间切分对比原主表特征与本次位置特征的验证 RMSE。此脚本只生成特征，不训练模型，也不声明已改善 RMSE。
