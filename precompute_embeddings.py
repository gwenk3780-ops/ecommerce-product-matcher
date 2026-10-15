"""
离线预计算站内所有商品的 CLIP 图像嵌入向量。
支持断点续跑：已处理的商品不会重复计算。
运行完成后生成 internal_embeddings.npz，供 match_items.py 直接加载。
"""

import os
import hashlib
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel

warnings.filterwarnings('ignore')

# ── 配置 ──────────────────────────────────────────────────────────────────────
CACHE_DIR       = 'image_cache'
CHECKPOINT_FILE = 'internal_embeddings_checkpoint.pkl'  # 断点文件
OUTPUT_FILE     = 'internal_embeddings.npz'             # 最终输出
CLIP_BATCH       = 64   # CLIP 推理批大小
DOWNLOAD_WORKERS = 64   # 并发下载线程数
CHECKPOINT_EVERY = 500  # 每处理 N 张保存一次断点

os.makedirs(CACHE_DIR, exist_ok=True)

# ── 1. 读取站内商品 ────────────────────────────────────────────────────────────
print("读取站内商品数据...")
internal = pd.read_excel('站内男装总表.xlsx', sheet_name='Sheet1')
internal = internal.copy()
internal['item_id']         = internal['item_id'].astype(str).str.strip()
internal['first_image_url'] = internal['first_image_url'].fillna('').str.strip()

# 只处理有图片 URL 的商品
df = internal[internal['first_image_url'] != ''][['item_id', 'first_image_url']].drop_duplicates('first_image_url')
print(f"有图片的商品: {len(df):,} / {len(internal):,}")

# ── 2. 加载 CLIP 模型 ──────────────────────────────────────────────────────────
print("\n加载 CLIP 模型...")
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
model     = CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device)
processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
model.eval()
print(f"CLIP 已加载，设备: {device}")

# ── 3. 加载断点（优先断点文件，其次已有索引） ──────────────────────────────────
if os.path.exists(CHECKPOINT_FILE):
    with open(CHECKPOINT_FILE, 'rb') as f:
        checkpoint = pickle.load(f)
    done_urls   = set(checkpoint['urls'])
    embeddings  = checkpoint['embeddings']   # list of np.ndarray
    saved_urls  = checkpoint['urls']         # list of str
    print(f"断点加载：已完成 {len(done_urls):,} 张")
elif os.path.exists(OUTPUT_FILE):
    idx = np.load(OUTPUT_FILE, allow_pickle=True)
    saved_urls  = idx['urls'].tolist()
    embeddings  = list(idx['embeddings'])
    done_urls   = set(saved_urls)
    print(f"从已有索引恢复：已完成 {len(done_urls):,} 张")
else:
    done_urls  = set()
    embeddings = []
    saved_urls = []
    print("未找到断点，从头开始")

# ── 4. 工具函数 ────────────────────────────────────────────────────────────────
def _cache_path(url: str) -> str:
    return os.path.join(CACHE_DIR, hashlib.md5(url.encode()).hexdigest() + '.jpg')

_session = requests.Session()
_session.headers.update({'User-Agent': 'Mozilla/5.0'})
adapter = requests.adapters.HTTPAdapter(
    pool_connections=DOWNLOAD_WORKERS,
    pool_maxsize=DOWNLOAD_WORKERS,
)
_session.mount('http://', adapter)
_session.mount('https://', adapter)

def download_image(url: str):
    """下载并缓存图片，返回 PIL Image 或 None。"""
    if not url:
        return None, url
    path = _cache_path(url)
    if os.path.exists(path):
        try:
            return Image.open(path).convert('RGB'), url
        except Exception:
            pass
    try:
        resp = _session.get(url, timeout=6)
        img = Image.open(BytesIO(resp.content)).convert('RGB')
        img.save(path, quality=85)
        return img, url
    except Exception:
        return None, url

def compute_embeddings(images: list) -> np.ndarray:
    """批量计算 CLIP 图像嵌入，返回 (N, 512) float32。"""
    inputs = processor(images=images, return_tensors='pt').to(device)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    result = feats.cpu().float().numpy()
    # 清空 MPS 缓存，防止显存碎片化导致进程被静默杀掉
    if device == 'mps':
        torch.mps.empty_cache()
    return result

def save_checkpoint():
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump({'urls': saved_urls, 'embeddings': embeddings}, f)

# ── 5. 主循环：并发下载 + 批量推理 ────────────────────────────────────────────
todo_urls = [u for u in df['first_image_url'].tolist() if u not in done_urls]
total_todo = len(todo_urls)
print(f"\n待处理: {total_todo:,} 张（跳过已完成 {len(done_urls):,} 张）")

if total_todo == 0:
    print("全部已完成，直接生成输出文件。")
else:
    buffer_imgs = []   # 待推理的图片
    buffer_urls = []   # 对应的 URL
    processed   = 0
    failed      = 0

    SUBMIT_BATCH = 512   # 每次只提交这么多任务，控制内存占用
    print(f"开始并发下载（{DOWNLOAD_WORKERS} 线程，每批提交 {SUBMIT_BATCH} 个任务）...\n")

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as executor:
        for batch_start in range(0, len(todo_urls), SUBMIT_BATCH):
            batch_urls = todo_urls[batch_start:batch_start + SUBMIT_BATCH]
            futures = {executor.submit(download_image, u): u for u in batch_urls}

            for future in as_completed(futures):
                img, url = future.result()
                if img is not None:
                    buffer_imgs.append(img)
                    buffer_urls.append(url)
                else:
                    failed += 1

                # 凑满一批就做 CLIP 推理
                if len(buffer_imgs) >= CLIP_BATCH:
                    vecs = compute_embeddings(buffer_imgs[:CLIP_BATCH])
                    for v, u in zip(vecs, buffer_urls[:CLIP_BATCH]):
                        embeddings.append(v)
                        saved_urls.append(u)
                    buffer_imgs = buffer_imgs[CLIP_BATCH:]
                    buffer_urls = buffer_urls[CLIP_BATCH:]
                    processed += CLIP_BATCH

                    # 定期保存断点
                    if processed % CHECKPOINT_EVERY < CLIP_BATCH:
                        save_checkpoint()
                        total_done = len(done_urls) + processed
                        pct = total_done / len(df) * 100
                        print(f"  进度: {total_done:,}/{len(df):,} ({pct:.1f}%)  失败: {failed}")

    # 处理剩余不足一批的图片
    if buffer_imgs:
        vecs = compute_embeddings(buffer_imgs)
        for v, u in zip(vecs, buffer_urls):
            embeddings.append(v)
            saved_urls.append(u)

    save_checkpoint()
    print(f"\n下载完成：成功 {len(saved_urls):,}，失败 {failed}")

# ── 6. 保存最终索引 ────────────────────────────────────────────────────────────
print(f"\n保存索引到 {OUTPUT_FILE}...")
emb_matrix = np.stack(embeddings).astype(np.float32)   # (N, 512)
np.savez_compressed(
    OUTPUT_FILE,
    embeddings = emb_matrix,
    urls       = np.array(saved_urls),
)
print(f"完成！索引大小: {emb_matrix.shape}，文件: {os.path.getsize(OUTPUT_FILE)/1024/1024:.1f} MB")

# 清理断点文件
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print("断点文件已清理")
