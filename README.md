# Item Recognition · 商品同品识别

面向电商商品检索的多模态匹配工具：结合商品标题和图片，在本地商品库中检索相似商品，并生成可复核的匹配结果。

## 核心能力

- 使用 CLIP 提取商品图像向量，结合 TF-IDF 标题相似度进行排序。
- 按上衣、下装、内衣分类筛选候选商品；Streamlit 版默认文本权重 10%、图像权重 90%，返回 Top 3。
- 离线预计算商品图像索引，支持图片缓存和断点续算。
- 上传 Excel 后交互式匹配，导出 Excel 和 HTML 预览。

## 运行

建议在 Python 3.10 或更高版本的独立虚拟环境中安装依赖。首次运行需联网下载 CLIP 模型，原实现支持 Apple MPS 或 CPU。

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python precompute_embeddings.py
python -m streamlit run app.py
```

运行前，自行准备 `站内男装总表.xlsx`（工作表名 `Sheet1`）。字段如下：

| 字段 | 用途 |
| --- | --- |
| item_id | 商品 ID |
| item_name | 商品名称 |
| merchant_name | 店铺名称 |
| sku_price | 价格 |
| item_url | 商品链接 |
| second_category_name | 商品类目 |
| active_status | 商品状态 |
| first_image_url | 可访问的商品图片 URL |

上传的 TK Excel 使用 `上衣`、`下装`、`内衣`工作表，支持 `商品名称`、`商品图片链接`（或 `商品图片`）、`商品售价`、`店铺名`等列。类目映射可在 `app.py` 的 `CATEGORY_MAP` 中调整。

`precompute_embeddings.py` 默认读取项目根目录的商品表并生成 `internal_embeddings.npz`。如需自定义 Web 应用的数据路径，可复制 `config.example.json` 为 `config.json` 后修改；预计算脚本的路径需同步修改。

## 文件说明

| 文件 | 职责 |
| --- | --- |
| app.py | Streamlit 交互式应用 |
| precompute_embeddings.py | 离线图像向量索引 |
| match_items.py | 文本与图像批处理匹配 |
| match_items_text_only.py | 纯文本匹配基线 |

批处理脚本保留原来的输入文件名，使用前需在脚本配置区调整为自己的文件名。

## 展示范围

仓库包含程序源码和运行说明，不包含业务商品表、商品图像缓存、预计算向量、真实匹配结果及内部部署配置。需要自行准备数据；本次发布只验证了 Python 语法，未执行真实数据匹配或模型效果评测。
