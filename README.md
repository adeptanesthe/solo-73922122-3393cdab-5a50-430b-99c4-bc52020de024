# 风机耗电共用模型平台（Fan Power Model Platform）

各厂只上报工况摘要（样本数 + 整数向量和 + 外积和 + 压差零点修正量），平台用
NumPy 岭回归拟合一套共用的"转速/压差 → 每分钟耗电量"模型，按留一厂交叉验证
选 λ，SQLite 多版本保存，支持凭版号原子替换单厂修正量。

支持 **3–256 家厂**建版。留一厂训练的工作量只随厂数线性增长。

## 快速开始

```bash
docker compose up --build      # 服务监听 :8000
./demo.sh                      # 演示：建厂 → 预测 → 更正修正量 → 冲突
                               #       → 连续更正让一折压差变常量
```

本地开发（无 Docker）：

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m uvicorn --factory app.main:create_app --port 8000
.venv/bin/python -m pytest tests/ -q     # 44 个自测
```

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/versions` | 上报 3–256 家厂的摘要，训练并发布新版本（201） |
| GET | `/versions/current` | 现行版：所选 λ、逐厂误差、修正量、模型、训练诊断 |
| GET | `/versions/{id}` | 历史版本（不可变） |
| POST | `/versions/{id}/correction` | 凭版号替换一厂修正量；成功则修正量+模型+逐厂误差作为新版本一起生效；版号冲突 409 / 重算失败 422，均保留现行版 |
| POST | `/predict` | 按厂号 + 原始转速、压差预测耗电，返回所用版号；重算期间沿用现行版 |
| GET | `/health` | 健康检查 |

### 上报格式（`POST /versions`）

```json
{
  "factories": [
    {
      "factory_id": "F1",
      "n": 1440,
      "sums":  [Σ转速, Σ压差, Σ耗电],
      "outer": [[Σs², Σsp, Σsy], [Σsp, Σp², Σpy], [Σsy, Σpy, Σy²]],
      "correction": 0
    }
  ]
}
```

`sums`/`outer` 均为**整数**（任意精度，JSON 大整数无损解析）；`correction`
为加到压差上的整数零点修正量。预测时传原始读数，平台自动加上该厂修正量。

### 256 厂建版与零点更正示例

```bash
# 生成确定性的 256 厂建版请求体（demo/create_version_256.json）
python3 scripts/make_256_demo.py

# 建版（实测约 0.13 s）
curl -sS -X POST http://localhost:8000/versions \
  -H 'Content-Type: application/json' \
  --data @demo/create_version_256.json

# 响应里的 training 块（摘要处理次数 = 2K-1 = 511；旧法逐折重合并要 K(K-1) = 65280 次）
# {
#   "version": 1, "lambda": 0.01,
#   "lambda_scores": {"0.01": ..., "0.1": ..., "1.0": ...},
#   "factory_errors": {"F001": ..., ... 共 256 项},
#   "training": {
#     "summary_ops": 511,
#     "legacy_summary_ops": 65280,
#     "matrix_factorizations": 256,
#     "near_tie_recheck": null,
#     "folds": [{"factory_id": "F001", "n_held_out": 120,
#                "kept": [0, 1], "dropped_features": []}, ...]
#   }
# }

# 凭现行版号做零点更正；冲突（版号过期）返回 409，重算失败返回 422，均不发布
curl -sS -X POST http://localhost:8000/versions/1/correction \
  -H 'Content-Type: application/json' \
  -d '{"factory_id": "F008", "correction": 40}'
```

更正让某一折压差变常量的完整 API 演练：

```bash
python3 scripts/demo_constant_pressure.py
# 连续三次更正后，留出 F3（唯一压差有波动的厂）的那一折训练池里
# 修正后压差全部等于 500000 -> 该折 dropped_features=["pressure"]，
# 其余各折仍保留两个特征。
```

## 关键设计

- **线性工作量的留一厂摘要**：先一次合并出全厂总计（K−1 次合并），每一折
  再用精确整数减法 `stats.remove(total, factory_k)` 得到其余厂的池化摘要
  （K 次减法），合计 **2K−1 次摘要处理**，与 λ 个数无关；旧实现每折重新
  合并 K−1 个厂需要 K(K−1) 次。减法与重合并逐位相等（有专门测试）。
- **每折至多一次矩阵分解**：每折对标准化后的相关阵 C 做一次对称特征分解
  `C = Q diag(d) Qᵀ`，三个候选 λ 共用同一组特征向量，
  `(C+λI)⁻¹c = Q diag(1/(d_j+λ)) Qᵀc` 只做矩阵-向量乘。因此三个 λ 合计
  每折最多一次分解（256 厂 = 256 次，而不是 768 次求解）。
- **舍入复核**：当某候选 λ 的最大留厂 MSE 与当前最优的相对差距 ≤ 1e-12 时，
  用独立的直接 LU 求解（与特征分解不同的算法）单独重算这些接近的候选，
  避免选参被浮点舍入翻转；复核记录保存在 `training.near_tie_recheck`。
  无接近情况时不触发。
- **逐折常量特征**：某一折训练池里转速或修正后压差为常量（池化方差为 0）
  时，只在该折忽略该特征；其他折不受影响，每折的取舍在
  `training.folds[].dropped_features` 里可见。
- **大基值方差保留**：协方差分子 `N·Σxxᵀ − ΣxΣxᵀ` 全程用 Python 大整数精确
  计算，最后一步才转 float64。读数基值 1e12、方差仅个位数时，朴素
  `E[x²]−E[x]²` 会被灾难性抵消成 0，这里仍精确保留（有专门测试）——即使
  被留出的是样本最多的厂，其余厂的非零方差也不会丢。
  入库前对所有厂做精确 PSD 校验（全部主子式 ≥ 0，整数判定）。
- **岭回归**：损失 = 均方误差 + λ·‖w‖²，截距免罚。特征按训练厂的 pooled
  均值与**总体标准差**标准化后，闭式解 `(C + λI)w = Cov(z,y)`，
  `b = ȳ`；零方差特征忽略。
- **留一厂验证**：轮流留一厂，只用其余厂的均值/总体标准差标准化并拟合，
  评估留出厂 MSE（由充分统计量精确求出，无需原始行）。在
  {0.01, 0.1, 1} 中选"各留出厂 MSE 最大值"最小的 λ，并列取较小者，
  再用全厂重训。快速特征分解路径与独立的逐折 LU 参考实现逐厂核对，
  各候选最大留厂 MSE、胜出 λ 的逐厂误差、全厂重训预测仅有浮点舍入差异
  （256 厂有专门等价性测试）。
- **版本原子性**：修正量、模型、逐厂误差、训练诊断在同一 SQLite 事务写入，
  最后才翻动 `current_version` 指针；预测只读已提交的现行版，因此重算期间
  预测仍沿用现行版。凭版号更新时若版号已过期 → 409 且现行版不变；
  重算失败（如更正后统计量不合法）→ 422 且同样不发布。
- **并发**：WAL 模式 + 写锁；计算在事务外完成，提交窗口极短。
- **旧库兼容**：启动时自动给旧版 `versions` 表补 `training_json` 列，
  历史版本的 `training` 块为空对象。

## 演示（demo.sh）

`scripts/make_demo_data.py` 生成确定性的 4 厂数据（转速基值 1e6、压差基值
5e5 的原始计数；F3 压差零点漂移 +500 但按 0 上报；F1/F2/F4 各自压差读数
恒定）。`demo.sh` 依次：

1. `POST /versions` 建厂 → 打印摘要处理次数（7 次；旧法 12 次）与逐折诊断；
2. `POST /predict` 用原始读数预测 F3；
3. `POST /versions/1/correction` 把 F3 修正量改为 500 → 新版本，
   F3 留厂误差从约 32 降到约 1.3，模型与修正量同时生效；
4. 再次预测 → 结果改变且 `version` 前进；
5. 用过期版号重放 → HTTP 409，现行版保持不变；
6. `scripts/demo_constant_pressure.py` 连续更正 F2/F4 的修正量，
   使留出 F3 的那一折修正后压差成为同一常量 → 该折单独忽略压差特征。

另有 `scripts/make_256_demo.py` 生成 256 厂请求体，配合上面的 curl 示例
演示上限建版（511 次摘要处理、256 次矩阵分解）。

## 剩余事项

- 本环境无 Docker，`docker compose` 未实际构建过；Dockerfile 为标准
  python:3.12-slim 流程，建议在有 Docker 的机器上跑一次 `docker compose up --build`。
- 单进程服务（uvicorn 单 worker）；多副本部署需把 SQLite 换成集中式存储。
- 未做鉴权/审计；生产环境应加认证与操作日志。
- 留一厂验证对工况迥异的厂会外推放大误差（属预期行为）；如需可扩展分组验证。
