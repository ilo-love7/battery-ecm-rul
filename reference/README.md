# reference/

## `LoadData.m`

MIT-Stanford 电池数据集（Severson et al., *Nature Energy* 2019）随数据发布的官方 MATLAB 脚本，
用于把 3 个 batch 文件整理成论文使用的 124 颗电池白名单。

**本仓库收录它的唯一目的**：作为数据筛选口径的对照参考。
`src/parsers/mit.py` 中的 `build_official_cell_list()` 逐条复刻了它的处理顺序
（batch2 续测拼接、batch3 异常剔除、未跑完电池剔除），以便任何人可以核对
「41 / 43 / 40 这 124 颗是怎么来的」。

**版权**：该脚本版权归数据集原作者所有，不属于本项目，亦不在本仓库的 MIT 许可范围内。
如需在其它场景使用，请遵循数据集原始许可条款。
