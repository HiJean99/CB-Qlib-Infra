# QLIB: 可转债量化数据基建

这是一个基于 Qlib 格式的可转债 (Convertible Bond) 数据采集、清洗与转换工程。它负责从 Tushare 获取全量的行情与特有因子数据，并将其转化为 Qlib 高效的二进制格式。

## 快速开始

### 1. 环境准备
确保拥有 `miniconda3` 并创建了 `q_lab` 环境：
```bash
conda create -n q_lab python=3.10
conda activate q_lab
pip install tushare pandas qlib loguru fire tqdm datacompy
```

### 2. 配置 API Token (极其重要)
为了安全起见，Token 不应存储在代码中。请按照以下步骤配置：

1.  在根目录下创建 `.env` 文件：
    ```bash
    cp .env.example .env
    ```
2.  编辑 `.env` 文件，填入你的 Tushare Token：
    ```text
    TUSHARE_TOKEN=你的_TUSHARE_TOKEN_字符串
    ```

### 3. 数据处理流程

-   **全量初始化**：运行 `./run_pipeline.sh` 获取历史至今的全量 CSV 数据。
-   **日常增量更新**：运行 `./run_daily_update.sh --start_date YYYYMMDD` 获取最新数据并追加到 CSV 中。
-   **转换为 Qlib 格式**：运行 `./run_dump.sh` 将清理好的 CSV 转换为 `~/.qlib/qlib_data/cb_data` 下的二进制文件并自动对账校验。

## 项目结构
- `scripts/`: 核心 Python 处理脚本。
- `csv_data/`: 存储原始 CSV 数据（不提交）。
- `run_*.sh`: 一键式操作脚本。

