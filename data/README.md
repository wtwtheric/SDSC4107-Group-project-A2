# data/

原始与中间数据不进 git（spec §7）。目录约定见 [docs/spec.md](../docs/spec.md) §2：

- `raw/<source>/<name>_<YYYYMMDD>.csv`：抓取当日的原始文件，文件名带抓取日期
- `interim/`：清洗中间件
- `processed/hk_daily.csv`：B1 产出的对齐后日频表

抓取脚本是 `scripts/fetch_hk.py`（B1 交付），原始 csv 不提交、不再分发。
