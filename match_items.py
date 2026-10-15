"""
TK热销品 × 站内同品匹配
文本:图像 = 3:7，图像向量来自预计算索引 internal_embeddings.npz
运行前请先执行 precompute_embeddings.py 生成索引。
"""

import os
import hashlib
import warnings
import json

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel

warnings.filterwarnings('ignore')

# ── 配置 ──────────────────────────────────────────────────────────────────────
TOP_N        = 3
TEXT_WEIGHT  = 0.15
IMAGE_WEIGHT = 0.85
CACHE_DIR    = 'image_cache'
INDEX_FILE   = 'internal_embeddings.npz'

os.makedirs(CACHE_DIR, exist_ok=True)

# ── 1. 读取数据 ──────────────────────────────────────────────────────────────
import time as _time
_t0 = _time.time()
print("读取文件...")
tk_sheets = pd.read_excel('26W12_tk_top500.xlsx', sheet_name=None)
internal  = pd.read_excel('站内男装总表.xlsx', sheet_name='Sheet1')

internal = internal.copy()
internal['item_name']            = internal['item_name'].fillna('').str.strip()
internal['merchant_name']        = internal['merchant_name'].fillna('')
internal['sku_price']            = internal['sku_price'].fillna('')
internal['item_url']             = internal['item_url'].fillna('')
internal['second_category_name'] = internal['second_category_name'].fillna('')
internal['active_status']        = internal['active_status'].fillna('')
internal['first_image_url']      = internal['first_image_url'].fillna('')
internal['item_id']              = internal['item_id'].fillna('').astype(str)

print(f"站内商品总数（正常）: {len(internal)}")

# ── 2. 类目映射 ──────────────────────────────────────────────────────────────
CATEGORY_MAP = {
    '上衣': ['上衣', '帽衫&运动衫', '毛衫& 开衫', '夹克,外套&马甲', '西装', '套装', '职业装', '传统服饰'],
    '下装': ['短裤', '裤装', '牛仔裤'],
    '内衣': ['贴身衣物&内衣', '睡衣', '袜子'],
}

# ── 3. 合并TK商品 ────────────────────────────────────────────────────────────
tk_rows = []
for sheet_name, df in tk_sheets.items():
    img_col = '商品图片链接' if '商品图片链接' in df.columns else '商品图片'
    for _, row in df.iterrows():
        tk_rows.append({
            'TK分类':     sheet_name,
            'TK商品名称': str(row.get('商品名称', '')).strip(),
            'TK图片':     str(row.get(img_col, '')).strip(),
            'TK价格':     str(row.get('商品售价', '')).strip(),
            'TK店铺':     str(row.get('店铺名', '')).strip(),
            '站内是否有同品': str(row.get('站内是否有同品', '')).strip(),
        })

tk_df = pd.DataFrame(tk_rows)
print(f"TK商品总数: {len(tk_df)}")

# ── 4. 加载预计算索引 ────────────────────────────────────────────────────────
if not os.path.exists(INDEX_FILE):
    raise FileNotFoundError(
        f"找不到 {INDEX_FILE}，请先运行 precompute_embeddings.py 生成索引。"
    )

print(f"\n加载图像索引 {INDEX_FILE}...")
idx_data   = np.load(INDEX_FILE, allow_pickle=True)
idx_embeds = idx_data['embeddings']   # (N, 512) float32
idx_urls   = idx_data['urls'].tolist()
url_to_vec = {u: idx_embeds[i] for i, u in enumerate(idx_urls)}
print(f"索引已加载：{len(idx_urls):,} 张图片")

# ── 5. 加载 CLIP 模型（只用于 TK 商品图片） ──────────────────────────────────
print("\n加载 CLIP 模型...")
device    = 'mps' if torch.backends.mps.is_available() else 'cpu'
clip_model     = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
clip_model.eval()
print(f"CLIP 已加载，设备: {device}")

# ── 6. 工具函数 ──────────────────────────────────────────────────────────────
def _cache_path(url: str) -> str:
    return os.path.join(CACHE_DIR, hashlib.md5(url.encode()).hexdigest() + '.jpg')

def download_image(url: str):
    if not url or url in ('', 'nan', 'None'):
        return None
    path = _cache_path(url)
    if os.path.exists(path):
        try:
            return Image.open(path).convert('RGB')
        except Exception:
            pass
    try:
        resp = requests.get(url, timeout=8,
                            headers={'User-Agent': 'Mozilla/5.0'})
        img = Image.open(BytesIO(resp.content)).convert('RGB')
        img.save(path, quality=85)
        return img
    except Exception:
        return None

def encode_images(urls: list[str]) -> dict:
    """下载并计算 TK 商品图片的 CLIP 向量（数量少，无需并发）。"""
    result = {}
    todo   = [u for u in urls if u and u not in url_to_vec]
    if todo:
        print(f"  下载 TK 商品图片: {len(todo)} 张...")
        for i in range(0, len(todo), 32):
            batch_urls = todo[i:i+32]
            imgs, valid = [], []
            for u in batch_urls:
                img = download_image(u)
                if img:
                    imgs.append(img)
                    valid.append(u)
            if imgs:
                inputs = clip_processor(images=imgs, return_tensors='pt').to(device)
                with torch.no_grad():
                    feats = clip_model.get_image_features(**inputs)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                for u, v in zip(valid, feats.cpu().float().numpy()):
                    result[u] = v
    for u in urls:
        if u in url_to_vec:
            result[u] = url_to_vec[u]
        elif u not in result:
            result[u] = None
    return result

def img_sim(v1, v2):
    if v1 is None or v2 is None:
        return None
    return float((np.dot(v1, v2) + 1) / 2)

# ── 7. 按类目匹配 ────────────────────────────────────────────────────────────
results = []

for category, internal_cats in CATEGORY_MAP.items():
    tk_sub       = tk_df[tk_df['TK分类'] == category].copy()
    if tk_sub.empty:
        continue

    internal_sub = internal[internal['second_category_name'].isin(internal_cats)].copy()
    print(f"\n[{category}] TK商品: {len(tk_sub)}, 站内候选: {len(internal_sub)}")

    if internal_sub.empty:
        for _, row in tk_sub.iterrows():
            r = row.to_dict()
            for i in range(1, TOP_N + 1):
                for key in ['id', '名称', '图片', '店铺', '价格', '链接', '状态', '相似度']:
                    r[f'同品{i}_{key}'] = ''
            results.append(r)
        continue

    internal_sub_reset = internal_sub.reset_index(drop=True)

    # ── Stage 1: TF-IDF 文本相似度矩阵 ───────────────────────────────────────
    all_names    = list(tk_sub['TK商品名称']) + list(internal_sub['item_name'])
    vectorizer   = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4),
                                   min_df=1, sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform([n.lower() for n in all_names])
    tk_vecs      = tfidf_matrix[:len(tk_sub)]
    int_vecs     = tfidf_matrix[len(tk_sub):]
    text_sims    = cosine_similarity(tk_vecs, int_vecs)   # (T, I)

    # ── Stage 2: 获取 TK 商品图像向量 ────────────────────────────────────────
    tk_url_list = tk_sub['TK图片'].tolist()
    tk_embeds   = encode_images(tk_url_list)

    # ── Stage 3: 加权打分，对全量候选排序 ────────────────────────────────────
    # 先构建站内候选的图像向量矩阵（从索引直接取，无需下载）
    int_urls   = internal_sub_reset['first_image_url'].tolist()
    int_vecs_img = np.array([
        url_to_vec.get(u, np.zeros(512, dtype=np.float32))
        for u in int_urls
    ], dtype=np.float32)   # (I, 512)

    # 标记哪些站内商品有有效向量
    has_img = np.array([url_to_vec.get(u) is not None for u in int_urls])

    for idx, (_, row) in enumerate(tk_sub.iterrows()):
        t_sims   = text_sims[idx]                  # (I,)
        tk_embed = tk_embeds.get(row['TK图片'])

        if tk_embed is not None:
            # 矩阵点积一次算完所有图像相似度
            img_sims_raw = int_vecs_img @ tk_embed  # (I,)
            img_sims = (img_sims_raw + 1) / 2       # 归一化到 [0,1]
            img_sims[~has_img] = 0.0                # 没有图片的设为 0

            # 有图片用加权，没图片退化为纯文本
            combined = np.where(
                has_img,
                TEXT_WEIGHT * t_sims + IMAGE_WEIGHT * img_sims,
                t_sims
            )
        else:
            combined = t_sims

        top_indices = np.argsort(combined)[::-1][:TOP_N]

        r = row.to_dict()
        for rank, i in enumerate(top_indices, 1):
            match = internal_sub_reset.iloc[i]
            r[f'同品{rank}_id']     = match['item_id']
            r[f'同品{rank}_名称']   = match['item_name']
            r[f'同品{rank}_图片']   = match['first_image_url']
            r[f'同品{rank}_店铺']   = match['merchant_name']
            r[f'同品{rank}_价格']   = match['sku_price']
            r[f'同品{rank}_链接']   = match['item_url']
            r[f'同品{rank}_状态']   = match['active_status']
            r[f'同品{rank}_相似度'] = round(float(combined[i]), 4)

        results.append(r)

result_df = pd.DataFrame(results)

# ── 8. 输出 Excel ────────────────────────────────────────────────────────────
weight_tag  = f'{TEXT_WEIGHT*10:g}_{IMAGE_WEIGHT*10:g}'
output_path = f'TK热销品站内同品匹配_{weight_tag}.xlsx'

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

wb = Workbook()
wb.remove(wb.active)

HEADER_FILL = PatternFill('solid', start_color='1F4E79')
HEADER_FONT = Font(name='Arial', bold=True, color='FFFFFF', size=10)
DATA_FONT   = Font(name='Arial', size=10)
ALT_FILL    = PatternFill('solid', start_color='EBF3FB')
MATCH_FILLS = [
    PatternFill('solid', start_color='E2EFDA'),
    PatternFill('solid', start_color='FFF2CC'),
    PatternFill('solid', start_color='FCE4D6'),
]

col_order = ['TK分类', 'TK商品名称', 'TK图片', 'TK价格', 'TK店铺', '站内是否有同品']
for i in range(1, TOP_N + 1):
    col_order += [f'同品{i}_id', f'同品{i}_名称', f'同品{i}_图片', f'同品{i}_店铺',
                  f'同品{i}_价格', f'同品{i}_相似度', f'同品{i}_状态', f'同品{i}_链接']

COL_WIDTHS = {'TK分类': 8, 'TK商品名称': 45, 'TK图片': 15,
              'TK价格': 10, 'TK店铺': 20, '站内是否有同品': 12}
for i in range(1, TOP_N + 1):
    COL_WIDTHS.update({f'同品{i}_id': 15, f'同品{i}_名称': 40, f'同品{i}_图片': 15,
                       f'同品{i}_店铺': 20, f'同品{i}_价格': 10,
                       f'同品{i}_相似度': 8, f'同品{i}_状态': 10, f'同品{i}_链接': 20})

for category in CATEGORY_MAP.keys():
    sub = result_df[result_df['TK分类'] == category].copy()
    if sub.empty:
        continue

    ws      = wb.create_sheet(title=category)
    headers = [c for c in col_order if c in sub.columns]

    for ci, col in enumerate(headers, 1):
        cell           = ws.cell(row=1, column=ci, value=col)
        cell.font      = HEADER_FONT
        cell.fill      = HEADER_FILL
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)

    for ri, (_, row) in enumerate(sub.iterrows(), 2):
        is_alt = (ri % 2 == 0)
        for ci, col in enumerate(headers, 1):
            val = row.get(col, '')
            if pd.isna(val):
                val = ''
            cell           = ws.cell(row=ri, column=ci, value=val)
            cell.font      = DATA_FONT
            cell.alignment = Alignment(vertical='center', wrap_text=True)
            for rank in range(1, TOP_N + 1):
                if col.startswith(f'同品{rank}_'):
                    cell.fill = MATCH_FILLS[rank - 1]
                    break
            else:
                if is_alt:
                    cell.fill = ALT_FILL

    for ci, col in enumerate(headers, 1):
        ws.column_dimensions[get_column_letter(ci)].width = COL_WIDTHS.get(col, 15)
    ws.row_dimensions[1].height = 28
    ws.freeze_panes = 'A2'

wb.save(output_path)
print(f"\n完成！结果已保存到: {output_path}")
print(f"共处理 {len(result_df)} 个TK商品")

# ── 9. 同步生成 preview_data.json ─────────────────────────────────────────────
def local_img_url(url: str) -> str:
    """如果本地缓存存在就用本地路径，否则保留原始 URL。"""
    if not url:
        return url
    local = os.path.join(CACHE_DIR, hashlib.md5(url.encode()).hexdigest() + '.jpg')
    return ('image_cache/' + hashlib.md5(url.encode()).hexdigest() + '.jpg') if os.path.exists(local) else url

img_cols = ['TK图片'] + [f'同品{i}_图片' for i in range(1, TOP_N + 1)]

preview_data = {}
for category in CATEGORY_MAP.keys():
    sub = result_df[result_df['TK分类'] == category].copy().fillna('')
    for col in img_cols:
        if col in sub.columns:
            sub[col] = sub[col].apply(local_img_url)
    preview_data[category] = sub.to_dict(orient='records')

json_path = f'preview_data_{weight_tag}.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(preview_data, f, ensure_ascii=False)
print(f"{json_path} 已更新")
print(f"总耗时: {_time.time() - _t0:.1f} 秒")
