# Z哥少妇战法·补票战法·TePu战法 Python 实战

## 目录

* [项目简介](#项目简介)
* [快速上手](#快速上手)
* [参数说明](#参数说明)

  * [脚本参数](#脚本参数)
  * [内置策略参数](#内置策略参数)

    * [1. BBIKDJSelector（少妇战法）](#1-bbikdjselector少妇战法)
    * [2. BBIShortLongSelector（补票战法）](#2-bbishortlongselector补票战法)
    * [3. BreakoutVolumeKDJSelector（TePu 战法）](#3-breakoutvolumekdjselectortepu-战法)
* [项目结构](#项目结构)
* [免责声明](#免责声明)

---

## 项目简介

本仓库提供两个核心脚本：

| 名称                    | 作用                                                                                     |
| --------------------- | -------------------------------------------------------------------------------------- |
| **`fetch_kline.py`**  | 从 AkShare 实时快照筛选出总市值 ≥ 指定阈值且排除创业板的股票，抓取其历史日 K 线并保存为 CSV。支持多线程、增量更新和今日快照自动补齐。           |
| **`select_stock.py`** | 读取本地 CSV 行情，根据 `configs.json` 中定义的策略（Selector）进行批量选股，并把结果写入 `select_results.log` 和控制台。 |

默认内置三大选股策略（位于 `Selector.py`）：

* **BBIKDJSelector**（中文别名：**少妇战法**）
* **BBIShortLongSelector**（中文别名：**补票战法**）
* **BreakoutVolumeKDJSelector**（中文别名：**TePu 战法**）

---

## 快速上手

### 1. 安装依赖

```bash
# 建议 Python 3.10+，使用虚拟环境
pip install -r requirements.txt
```

> 依赖主要包括：`akshare`、`pandas`、`numpy`、`tqdm` 等。

### 2. 下载历史行情

```bash
python fetch_kline.py \
  --small-player True            # 不包含创业板数据
  --min-mktcap 2.5e10 \          # 市值阈值（默认 250 亿）
  --start 20050101 \             # 起始日期
  --end today \                  # 结束日期
  --out ./data \                 # 输出目录
  --workers 20                   # 并发线程数
```

### 3. 运行选股

```bash
python select_stock.py \
  --data-dir ./data \            # CSV 行情目录
  --config ./configs.json \      # 策略配置
  --date 2025-06-14              # 交易日（缺省=最新）
```

日志示例：

```
============== 选股结果 [TePu 战法] ===============
交易日: 2025-06-14
符合条件股票数: 1
600690
```

---

## 参数说明

### 脚本参数

| 脚本                | 关键参数              | 说明                                       |
| ----------------- | ----------------- | ---------------------------------------- |
| `fetch_kline.py`  | `--small-player`  | 是否包含创业板数据。设为 True 则不包含创业板数据              |
|                   | `--min-mktcap`    | 市值过滤阈值（元）。默认 2.5e10                      |
|                   | `--start / --end` | 日期范围，格式 `YYYYMMDD`；`--end today` 自动取当前日期 |
|                   | `--workers`       | 并发线程数，默认 20                              |
| `select_stock.py` | `--date`          | 选股所用交易日；缺省时自动取数据中最新日期                    |
|                   | `--tickers`       | `all` 或逗号分隔股票代码列表，精细控制股票池                |
|                   | `--config`        | Selector 配置文件路径，默认 `configs.json`        |

### 内置策略参数

> 以下参数来自 `configs.json`，可按需调整。

#### 1. BBIKDJSelector（少妇战法）

| 参数                | 示例值   | 说明                                                |
| ----------------- | ----- | ------------------------------------------------- |
| `threshold`       | `-6`  | 日线 **J 值上限**。当当天 J < threshold 时满足条件，阈值越低要求越严格。   |
| `bbi_min_window`  | `17`  | 用于检测 **BBI 单调上升** 的最短窗口长度（交易日数）。                  |
| `bbi_offset_n`    | `2`   | 选定距今日 *n* 日的锚点，可避免近期震荡。                           |
| `max_window`      | `60`  | 计算技术指标时最多读取的 K 线天数，限制窗口大小防止性能下降。                  |
| `price_range_pct` | `100` | 在最近 `max_window` 根 K 线上，收盘价高低波动幅度上限 (%)。限制妖股大起大落。 |

该战法核心：**BBI 持续上升** + **J 低位** + **DIF>0**，并附加“收盘价波动不过大”过滤，以剔除近期过度拉升的个股。

#### 2. BBIShortLongSelector（补票战法）

| 参数               | 示例值  | 说明                                          |
| ---------------- | ---- | ------------------------------------------- |
| `n_short`        | `3`  | **短期 RSV** 窗口 N1。滚动取近 N1 日最低 / 收盘价。         |
| `n_long`         | `21` | **长期 RSV** 窗口 N2。                           |
| `m`              | `3`  | **判别区间长度**。在最近 m 个交易日内同时满足 RSV、BBI、DIF 等条件。 |
| `bbi_min_window` | `5`  | BBI 上升段的最短窗口。                               |
| `bbi_offset_n`   | `0`  | BBI 锚点距当前日的偏移量。                             |
| `max_window`     | `60` | 读取历史 K 线最大长度。                               |

复刻经典“补票”逻辑：短长 RSV 同步高位 + BBI 上升 + DIF>0，并要求短 RSV 区间内曾跌破 20 形成“回抽”。

#### 3. BreakoutVolumeKDJSelector（TePu 战法）

| 参数                 | 示例值      | 说明                                                      |
| ------------------ | -------- | ------------------------------------------------------- |
| `j_threshold`      | `1`      | 观测日 T0 的 **J 值上限**（J < 此值）。                             |
| `up_threshold`     | `3.0`    | 在回溯窗口存在某日 T **较前一日涨幅** ≥ 此阈值 (%)，视为放量长阳。                |
| `volume_threshold` | `0.6667` | **缩量比例**。窗口内除 T 外所有成交量 ≤ `volume_threshold × volume_T`。 |
| `offset`           | `15`     | 放量突破回溯窗口大小（交易日数）。                                       |
| `max_window`       | `60`     | 参与指标计算的最大 K 线数。                                         |
| `price_range_pct`  | `100`    | 在最近 `max_window` 根 K 线上的收盘价高低波动幅度上限 (%)。                |

TePu 战法步骤：

1. 近期整体波动不剧烈（`price_range_pct` 过滤）。
2. 在 `offset` 窗口内寻找**单日放量长阳**（涨幅≥`up_threshold`、成交量最大）。
3. 此后成交量整体缩量，KDJ 指标维持高位，且当前 J < `j_threshold`、DIF>0 实现低吸。

---

## 项目结构

```
.
├── appendix.json            # 额外自选股票池（会与市值筛选结果合并）
├── configs.json             # Selector 运行时配置
├── fetch_kline.py           # 历史行情抓取脚本
├── select_stock.py          # 批量选股脚本
├── Selector.py              # 策略实现
├── data/                    # CSV 数据目录（运行后生成）
├── fetch.log
└── select_results.log
```

---

## 免责声明

* 本仓库代码仅供学习与技术研究之用，**不构成任何投资建议**。股市有风险，入市需谨慎。
* 感谢师尊 **@Zettaranc** [https://b23.tv/JxIOaNE](https://b23.tv/JxIOaNE) 的无私分享
