# pairwise-water-caustic-refract

## 领域

游戏、图形与多媒体

## 主题

水面折射与焦散

## 说明

当帧折射矩阵、焦散采样顺序与泡沫跟随。

初始状态下 `graphics/water_caustic_refract.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
