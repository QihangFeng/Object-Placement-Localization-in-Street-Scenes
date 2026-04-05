# 100 张可视化对比 + Best 10 精选

## 生成流程

### 模型与数据
- 模型: lr=1e-5 训练的 SupportSurface (SS) 和 HardConstraint (HC)
- 数据: Cityscapes val 集 500 张中随机抽取 100 张 (seed=1)
- Prompt: "place a person" 和 "place a car" 各一次
- 每张图输出格式: 左侧 top1 预测 + 右侧 top5 预测

### Vulcan 作业
在 Vulcan HPC (Alliance Canada) 上提交 2 个并行 SLURM 作业:

| 作业 | Job ID | Checkpoint | 参数 |
|------|--------|------------|------|
| SS | 4617212 | `models_lr1e5/02_supportsurface_best.pt` (epoch=3, top1=0.0592) | 无 `--use_hard_mask` |
| HC | 4617213 | `models_lr1e5/03_ss_hard_best.pt` | 加 `--use_hard_mask` |

- 分区: `gpubase_bygpu_b2` (12h wall time)
- GPU: 1x L40S, 16G 内存
- 两个作业使用相同 seed=1, 保证抽取的 100 张图完全一致, 文件名一一对应

### 生成结果
```
vis_supportsurface/    200 张 (100 图 x 2 prompts)
vis_hardconstraint/    200 张 (100 图 x 2 prompts)
共计 400 张 PNG 文件
```

文件命名: `{城市}_{编号}_{帧号}_place_a_{person|car}.png`

---

## Best 10 精选

从 100 组图片中审查 SS 与 HC 的配对差异, 按以下标准筛选:
1. SS vs HC 差异明显 (如 person 从马路移到人行道)
2. 预测位置视觉上合理
3. 场景清晰, 适合展示
4. "place a person" 和 "place a car" 预测有区分度

### 精选列表

| # | 图片 stem | 类别 | 入选理由 |
|---|----------|------|---------|
| 1 | lindau_000024_000019 | HC改进 | Person: SS 预测在路面中央, HC 明显偏移到右侧人行道。Car: HC 出现负分过滤。经典的"语义约束将行人从马路推向人行道"案例。 |
| 2 | lindau_000017_000019 | HC过滤 | Person+Car: HC 的 top2-5 得分全部为 -10000 (被 hard mask 完全过滤)。窄巷场景中, HC 过滤掉了几乎所有候选框, 展示 constraint 的极端行为和 tradeoff。 |
| 3 | lindau_000035_000019 | HC过滤 | Person+Car 均出现 -10000 分数。HC top1 仍在合理位置, 但 top2-5 被 hard mask 否决。展示过滤强度。 |
| 4 | munster_000002_000019 | HC改进 | Person: SS 在左侧路面放了大框, HC 缩小为右侧人行道小框。明显的路面到人行道偏移, 场景为林荫大道, 视觉清晰。 |
| 5 | munster_000025_000019 | HC改进 | Person: SS top1 在路面左侧, HC top1 偏向右侧人行道, top5 集中在人行道区域。宽阔街道+斑马线, 展示效果好。 |
| 6 | frankfurt_000000_008206 | HC改进 | 清晰的十字路口场景 (有行人、白车)。Person: SS top5 在路面分散, HC top5 更集中偏向人行道侧。适合展示城市场景泛化能力。 |
| 7 | frankfurt_000001_007973 | 场景丰富 | 斑马线+建筑街道场景。Person: HC top5 包含右侧人行道预测。Car: HC 出现负分过滤。同时展示 person vs car 的 prompt 区分度。 |
| 8 | frankfurt_000000_009291 | 场景丰富 | 有轨电车路口 (tram tracks)。Person: HC 将预测推离路面。Car: HC top5 出现负分。独特的交通场景, 丰富展示多样性。 |
| 9 | lindau_000034_000019 | HC过滤 | Person: HC 出现 -10000 分数, SS 正常分布。小镇街道场景, 展示 HC 在不同城市形态下的过滤效果。 |
| 10 | frankfurt_000000_007365 | 场景丰富 | 宽阔道路+绿化带+高楼。Person 和 Car 的 top5 分布 SS vs HC 有可见差异。场景开阔明亮, 适合报告展示。 |

### 精选分类

**HC 改进类 (#1, #4, #5, #6)**: 展示 HardConstraint 的核心价值 -- 通过语义 mask 将行人预测从路面约束到人行道, 提高 placement 的物理合理性。

**HC 过滤类 (#2, #3, #9)**: 展示 hard mask 的激进过滤 (-10000 分数), 说明 constraint 过强时会导致可用候选不足的 tradeoff。这解释了为什么 HC 的 IoU 指标 (0.0576) 低于 SS (0.0592) -- 过滤虽然提高了合理性, 但也排除了部分正确候选。

**场景丰富类 (#7, #8, #10)**: 补充多样性 -- 斑马线、有轨电车路口、宽阔道路等不同场景, 展示模型在多种城市环境下的泛化能力。

### 文件结构
```
selected-best-10/
  vis_supportsurface/    20 张 (10 stems x 2 prompts)
  vis_hardconstraint/    20 张 (10 stems x 2 prompts)
  共计 40 张文件
```

对比方式: 同名文件在 `vis_supportsurface/` 和 `vis_hardconstraint/` 中一一配对查看。
