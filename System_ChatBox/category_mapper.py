"""
category_mapper.py
Mô-đun RUNTIME — dùng chỉ mục nhỏ đã build sẵn bởi build_category_index.py
(chạy 1 lần offline) để ánh xạ category tiếng Việt do NeedExtractor trích
xuất (vd "thảm trải sàn", "nồi nấu", "ghế sofa") sang các leaf_category
TIẾNG ANH THẬT đang tồn tại trong bảng `products` (vd "Area Rugs",
"Stockpots"/"Saucepans"/"Dutch Ovens", "Sofas & Couches"...).

Vì sao trả về NHIỀU leaf_category thay vì chỉ 1:
  Một từ tiếng Việt như "nồi" rộng hơn 1 leaf_category tiếng Anh cụ thể —
  nó có thể tương ứng với nhiều leaf_category khác nhau (Stockpots,
  Saucepans, Dutch Ovens, Pressure Cookers...). Trả về top-N category gần
  nghĩa nhất rồi lọc SQL bằng `leaf_category IN (...)` cho kết quả ĐÚNG hơn
  là ép về đúng 1 category (dễ bỏ sót) hoặc LIKE trực tiếp câu tiếng Việt
  (không bao giờ khớp — đây chính là lỗi gốc đã phát hiện).

KHÔNG gọi thêm Qwen — chỉ dùng lại SentenceTransformer đã nạp sẵn (giống
vector_search.py), nên KHÔNG phá nguyên tắc "đúng 2 lần gọi Qwen/lượt chat"
đã thống nhất trong thiết kế.

Nếu chỉ mục category chưa được build (chưa chạy build_category_index.py)
hoặc không có match nào đủ tin cậy (dưới ngưỡng), trả về [] — RecommendationService
(rule_based_filter trong main.py) khi đó sẽ KHÔNG áp lọc cứng category (đúng
tinh thần "không tự nới hay tự đặt ràng buộc khi chưa đủ căn cứ" — mục 3.5.2
khoá luận — ở đây là chiều ngược lại: không tự BỎ QUA candidate chỉ vì chưa
map được tên, để AI Matching/free_text vẫn có cơ hội xử lý).
"""

import os
import re
import unicodedata
from typing import Dict, List, Optional

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

CATEGORY_DB_PATH = os.getenv("CATEGORY_DB_PATH", "./category_index_db")
COLLECTION_NAME = "leaf_categories"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")

# Ngưỡng cosine similarity tối thiểu để coi 1 leaf_category là match đủ tin cậy.
# 0.30 là điểm khởi đầu THẬN TRỌNG (chưa hiệu chỉnh bằng thực nghiệm) — nhóm
# NÊN tinh chỉnh giá trị này sau khi có vài chục câu hỏi thật để so sánh, đúng
# tinh thần "cấu hình... chốt trên phần phát triển trước đánh giá chính thức"
# (mục 3.5.3 khoá luận). Ghi lại quyết định cuối khi chốt.
DEFAULT_MIN_SCORE = 0.30
DEFAULT_TOP_N = 5

_model = None
_collection = None
_warned_unavailable = False

# Cache trong phiên chạy của tiến trình — nhiều user có thể hỏi cùng 1 category
# phổ biến ("nồi", "thảm"...), không cần encode lại mỗi lần.
_cache: dict = {}

# --- MỚI: bảng gợi ý từ khóa tiếng Việt cho các danh từ NGẮN/MƠ HỒ mà vector
# similarity của model đa ngôn ngữ đang dùng (paraphrase-multilingual-MiniLM-
# L12-v2) xử lý KÉM — đã phát hiện thật qua test: "nồi"/"nồi nấu"/"nồi nấu ăn"
# đều KHÔNG đưa "Stockpots" vào top-5, thậm chí "nồi" một mình còn khớp ra
# "Vacuums" (máy hút bụi, hoàn toàn không liên quan).
#
# KHÔNG dùng cách tự động tìm chuỗi con "pot" trong leaf_category — sẽ dính
# nhầm "Potato Mashers", "Pot Racks", "Potholders", "Potpourris" (đều chứa
# "pot" nhưng chẳng liên quan nồi nấu). Danh sách dưới đây đã CHỌN TAY, xác
# minh từng tên đúng là leaf_category THẬT tồn tại trong dataset v4 (đối
# chiếu bằng script khảo sát, không đoán).
#
# Cơ chế: nếu category text (đã bỏ dấu) CHỨA 1 từ khóa dưới đây (khớp
# NGUYÊN TỪ, không phải substring thô — cùng kỹ thuật word-boundary đã
# dùng trong color_mapper.py để tránh lỗi kiểu "khong biet" khớp nhầm
# "hồng"), GỘP THẲNG danh sách category tương ứng vào kết quả — KHÔNG phụ
# thuộc ngưỡng min_score, vì đây là ánh xạ đã xác minh tay, tin cậy hơn
# điểm vector. Vector search vẫn chạy thêm để bổ sung nếu thiếu, không bị
# thay thế hoàn toàn.
#
# Đây là danh sách KHỞI ĐẦU, chỉ phủ từ đã phát hiện lỗi qua test — nhóm
# nên tiếp tục bổ sung khi test thấy từ tiếng Việt khác cũng bị vector
# search xử lý sai, giống cách color_mapper.VI_TO_EN_COLOR đã phát triển
# dần qua thời gian.
VI_CATEGORY_HINTS: Dict[str, List[str]] = {
    "noi": ["Stockpots", "Pots & Pans", "Dutch Ovens", "Saucepans",
            "Pressure Cookers", "Slow Cookers", "Rice Cookers",
            "Multipots & Pasta Pots", "Hot Pots"],
}

_HINT_SORTED_KEYS = sorted(VI_CATEGORY_HINTS.keys(), key=len, reverse=True)


def _normalize_vi(text: str) -> str:
    """Bỏ dấu tiếng Việt + về chữ thường — dùng để so khớp VI_CATEGORY_HINTS
    không phân biệt hoa/thường, không phân biệt cách gõ dấu."""
    nfkd = unicodedata.normalize("NFKD", text.strip().lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _lookup_hints(vi_category_text: str) -> List[str]:
    """Trả về danh sách leaf_category từ VI_CATEGORY_HINTS nếu category text
    chứa 1 từ khóa đã biết (khớp NGUYÊN TỪ bằng \\b, không phải substring thô
    — tránh lỗi tương tự "khong biet" khớp nhầm "hồng" đã gặp ở color_mapper).
    Trả về [] nếu không khớp từ khóa nào."""
    normalized = _normalize_vi(vi_category_text)
    for key in _HINT_SORTED_KEYS:
        if re.search(rf"\b{re.escape(key)}\b", normalized):
            return VI_CATEGORY_HINTS[key]
    return []


def _lazy_init() -> bool:
    global _model, _collection, _warned_unavailable

    if _collection is not None:
        return True

    if SentenceTransformer is None or chromadb is None:
        if not _warned_unavailable:
            print("[category_mapper] Thiếu sentence-transformers/chromadb — "
                  "bỏ qua ánh xạ category, rule_based_filter sẽ không lọc cứng category.")
            _warned_unavailable = True
        return False

    try:
        client = chromadb.PersistentClient(path=CATEGORY_DB_PATH)
        _collection = client.get_collection(name=COLLECTION_NAME)
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return True
    except Exception as e:
        if not _warned_unavailable:
            print(f"[category_mapper] Chưa có chỉ mục category hợp lệ ({e}) — "
                  f"hãy chạy build_category_index.py trước. Tạm thời rule_based_filter "
                  f"sẽ không lọc cứng category.")
            _warned_unavailable = True
        return False


def resolve_leaf_categories(
    vi_category_text: str,
    top_n: int = DEFAULT_TOP_N,
    min_score: float = DEFAULT_MIN_SCORE,
) -> List[str]:
    """Trả về danh sách leaf_category TIẾNG ANH THẬT (đã tồn tại trong DB)
    gần nghĩa nhất với `vi_category_text` (tiếng Việt, do NeedExtractor trích
    xuất). Trả về [] nếu chưa sẵn sàng hoặc không có match nào đủ tin cậy —
    KHÔNG bao giờ bịa ra tên category không tồn tại.

    THỨ TỰ ưu tiên (MỚI):
      1. VI_CATEGORY_HINTS — bảng đã xác minh tay cho các từ mơ hồ mà vector
         search xử lý kém (xem giải thích ở khai báo VI_CATEGORY_HINTS phía
         trên). Nếu khớp, các category này LUÔN được đưa vào kết quả, KHÔNG
         phụ thuộc min_score.
      2. Vector search (như cũ) — chạy THÊM để bổ sung nếu còn thiếu, không
         thay thế hints. Nếu index chưa sẵn sàng nhưng đã có hint khớp, vẫn
         trả về đúng hint (không cần model)."""
    if not vi_category_text or not vi_category_text.strip():
        return []

    key = (vi_category_text.strip().lower(), top_n, min_score)
    if key in _cache:
        return _cache[key]

    matched: List[str] = list(_lookup_hints(vi_category_text))  # có thể rỗng

    if _lazy_init():
        try:
            query_vec = _model.encode(vi_category_text, normalize_embeddings=True)
            result = _collection.query(
                query_embeddings=[query_vec.tolist()],
                n_results=top_n,
            )
            ids = result.get("ids", [[]])[0]
            # Chroma mặc định trả "distances" (khoảng cách), KHÔNG phải
            # similarity. Với vector đã chuẩn hoá đơn vị và index cosine,
            # cosine_similarity = 1 - cosine_distance.
            distances = result.get("distances", [[]])[0]
            for cat_name, dist in zip(ids, distances):
                similarity = 1.0 - dist
                if similarity >= min_score and cat_name not in matched:
                    matched.append(cat_name)
        except Exception as e:
            print(f"[category_mapper] Lỗi truy vấn chỉ mục category: {e}")
            # Không return [] ở đây nữa — vẫn giữ kết quả từ hints (nếu có).

    _cache[key] = matched
    if not matched:
        print(f"[category_mapper] Không có leaf_category nào đủ tin cậy (>= {min_score}) "
              f"cho category='{vi_category_text}' — bỏ qua lọc cứng category ở lượt này.")
    return matched


if __name__ == "__main__":
    # Demo nhanh — cần đã chạy build_category_index.py trước.
    for demo in ["thảm trải sàn", "nồi nấu", "nồi nấu ăn", "nồi", "ghế sofa", "dao làm bếp", "gối ôm"]:
        print(demo, "->", resolve_leaf_categories(demo))