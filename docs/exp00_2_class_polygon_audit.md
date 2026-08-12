# Exp00.2 类别、实例、polygon 与尺度审计

状态：**PASS（审计完成），标注一致性 Gate：STOP**。

从全部 JSON 自动得到 7 类和 1847 个实例：Burn 426、Crack 140、Dent 202、Material missing 249、Tears 66、Tip curl 35、corrosion 729。最多/最少类实例比为 20.83:1，属于严重不均衡；corrosion 占 39.47%，Tip curl 仅占 1.89%。

程序未发现 polygon 越界、少于 3 个有效点、零面积、自交、NaN/Inf、重复 polygon、非 polygon shape 或 JSON/真实图片尺寸不一致。该结论表示几何与 schema 合法，不代表类别语义和 mask 边界已经人工验收。

296/1847（16.03%）实例的 polygon 相对面积小于 0.001。Dent 最突出：78/202（38.61%）属于该极小定义。全数据相对面积中位数 0.00584，q25/q50/q75 为 0.001616/0.005844/0.025067，建议后续采用这些数据分位数冻结 tiny/small/medium/large 边界。

