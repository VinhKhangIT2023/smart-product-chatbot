"""
vector_search.py
Mô-đun AI Matching THẬT (thay cho stub ai_matching_rank trong main.py) —
giai đoạn RETRIEVAL của RAG, chạy REAL-TIME mỗi lượt chat, nhưng KHÔNG tự
mã hóa lại toàn bộ catalog — chỉ đọc index đã build sẵn (build_vector_index.py)
và encode câu mô tả nhu cầu của user (1 câu, rất nhanh).

Chiến lược 2 tầng đã thống nhất trong khóa luận (mục 1.3.3, 3.5.3):
  Tầng 1 (rule_based_filter trong main.py): lọc cứng bằng SQL -> tập ứng
    viên nhỏ (vài chục sản phẩm).
  Tầng 2 (module này): trong tập ứng viên đã lọc, LẤY LẠI vector đã lưu
    sẵn của đúng những sản phẩm đó từ Chroma (bằng ID), tính cosine
    similarity với vector câu mô tả nhu cầu, sắp giảm dần -> Top-K.

Cách lấy lại vector theo ID (collection.get) thay vì query toàn bộ Chroma
rồi lọc sau — vì tập ứng viên đã nhỏ (rule-based lọc trước), lấy đúng ID
cần thiết sẽ nhanh và chính xác hơn dùng "where" filter phức tạp.

--- MỚI THÊM: extra_signals (đúng Bảng 2.9 khóa luận — brand/color/material
"chỉ lọc cứng khi bắt buộc") ---
Từ khi main.py phân biệt hard_constraint (lọc cứng SQL) với ưu tiên mềm cho
color/brand (xem need_extractor.py, session_manager.py), những giá trị
color/brand KHÔNG được lọc cứng ở tầng 1 cần một nơi khác để vẫn có ảnh
hưởng tới kết quả — đó là TẦNG 2 này, qua tham số extra_signals: main.py
truyền vào các giá trị color/brand đang ở chế độ "ưu tiên mềm" (chưa bị lọc
SQL), build_query_text() sẽ ghép chúng vào câu truy vấn ngữ nghĩa, giúp sản
phẩm có màu/thương hiệu gần đúng được XẾP HẠNG cao hơn, dù không bị LOẠI
nếu thiếu.
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import chromadb
except ImportError:
    chromadb = None

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = "products"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")

_model = None
_collection = None
_warned_unavailable = False


def _lazy_init() -> bool:
    """Nạp model + Chroma collection 1 lần duy nhất (cache), tái dùng cho
    mọi lượt chat sau đó — tránh load lại model mỗi request (rất chậm)."""
    global _model, _collection, _warned_unavailable

    if _collection is not None:
        return True

    if SentenceTransformer is None or chromadb is None:
        if not _warned_unavailable:
            print("[vector_search] Thiếu sentence-transformers/chromadb — AI Matching sẽ fallback về thứ tự gốc.")
            _warned_unavailable = True
        return False

    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = client.get_collection(name=COLLECTION_NAME)
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return True
    except Exception as e:
        if not _warned_unavailable:
            print(f"[vector_search] Chưa có index Chroma hợp lệ ({e}) — chạy build_vector_index.py trước.")
            _warned_unavailable = True
        return False


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def build_query_text(slots: Dict[str, Any], extra_signals: Optional[List[str]] = None) -> str:
    """Ghép các slot mềm (không phải ràng buộc cứng đã lọc SQL rồi) thành
    1 câu mô tả nhu cầu, dùng làm query cho vector search — chỉ nên đưa
    các thuộc tính NGỮ NGHĨA mơ hồ (style, material, free_text), vì
    category/price đã được rule_based_filter xử lý chính xác ở tầng 1 rồi
    (LUÔN hard), không cần lặp lại ở đây.

    extra_signals (MỚI): các giá trị color/brand đang ở chế độ ƯU TIÊN MỀM
    (KHÔNG bị lọc cứng SQL ở lượt này — xem main.py) — main.py tự quyết định
    và truyền vào đây, module này không tự biết field nào đã hard/soft."""
    parts = []
    for key in ("style", "material", "free_text"):
        value = slots.get(key)
        if value:
            parts.append(str(value))
    if extra_signals:
        parts.extend(str(s) for s in extra_signals if s)
    return " ".join(parts) if parts else (slots.get("category") or "")


def semantic_rank(
    candidates: List[Dict[str, Any]],
    slots: Dict[str, Any],
    top_k: int = 5,
    id_field: str = "product_id",
    extra_signals: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Xếp hạng lại `candidates` (đã qua rule_based_filter) theo độ phù hợp
    ngữ nghĩa với nhu cầu trong `slots` (+ extra_signals nếu có — xem
    build_query_text). Trả về top_k dict gốc từ `candidates`, có gắn thêm
    khoá "_similarity_score" để tham khảo/log.
    Nếu index chưa sẵn sàng, fallback: trả về top_k đầu tiên không đổi
    thứ tự (KHÔNG báo lỗi/crash toàn hệ thống — đúng nguyên tắc "không
    khẳng định đã chạy AI Matching khi chưa có vector hợp lệ", mục 3.5.3
    khóa luận)."""
    if not candidates:
        return []

    if not _lazy_init():
        return candidates[:top_k]

    query_text = build_query_text(slots, extra_signals)
    if not query_text.strip():
        return candidates[:top_k]

    ids = [str(c.get(id_field)) for c in candidates if c.get(id_field) is not None]
    if not ids:
        return candidates[:top_k]

    try:
        stored = _collection.get(ids=ids, include=["embeddings"])
    except Exception as e:
        print(f"[vector_search] Lỗi lấy vector từ Chroma: {e}")
        return candidates[:top_k]

    id_to_embedding = dict(zip(stored["ids"], stored["embeddings"]))
    query_vec = _model.encode(query_text, normalize_embeddings=True)

    scored = []
    for c in candidates:
        pid = str(c.get(id_field))
        vec = id_to_embedding.get(pid)
        if vec is None:
            continue  # sản phẩm này chưa được index (chưa chạy build_vector_index.py cho nó) -> bỏ qua, không bịa điểm
        score = _cosine_sim(np.array(vec), np.array(query_vec))
        enriched = dict(c)
        enriched["_similarity_score"] = round(score, 4)
        scored.append(enriched)

    if not scored:
        return candidates[:top_k]

    scored.sort(key=lambda x: x["_similarity_score"], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    # Demo nhanh — cần đã chạy build_vector_index.py trước.
    demo_candidates = [
        {"product_id": "TEST001", "title": "Thảm lông mềm màu be họa tiết hình học"},
    ]
    demo_slots = {"style": "tối giản, họa tiết hình học"}
    print(semantic_rank(demo_candidates, demo_slots))
    print(semantic_rank(demo_candidates, demo_slots, extra_signals=["xanh dương"]))