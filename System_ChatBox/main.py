"""
main.py
File chạy chính — FastAPI backend ghép các mô-đun (đúng Bảng 3.1 khóa luận):
  1. session_manager.py  -> DialogueService: decide_next_action() quyết định
     hành động mỗi lượt: ASK_HARD_SLOT / ASK_SOFT_SLOT / SEARCH_PRODUCTS / END_SESSION.
     MỚI: cũng lưu hard_constraint_slots (color/brand nào user nói RÕ là bắt buộc).
  2. need_extractor.py    -> NeedExtractor: trích xuất/cập nhật slot nhu cầu
     từ tin nhắn user (gọi Qwen riêng, tách khỏi bước sinh câu trả lời).
     MỚI: cũng trích "hard_constraints" (đúng Bảng 2.9 khóa luận).
  3. category_mapper.py / color_mapper.py -> lớp DỊCH thuộc tính tiếng Việt
     (do NeedExtractor trích xuất) sang giá trị tiếng Anh THẬT đang tồn tại
     trong catalog, TRƯỚC KHI đưa vào rule_based_filter().
  4. rule_based_filter()  -> RecommendationService tầng 1: lọc cứng bằng SQL
     trên bảng `products` (MySQL, đã import 15.714 sản phẩm).
     MỚI: category/size_space/price_range LUÔN lọc cứng như cũ; nhưng
     color/brand CHỈ lọc cứng nếu nằm trong hard_constraint_slots của phiên
     (đúng Bảng 2.9: "chỉ lọc cứng khi người dùng yêu cầu bắt buộc") — nếu
     không, để dành cho tầng 2 xử lý như ƯU TIÊN MỀM.
  5. vector_search.py     -> RecommendationService tầng 2 (AI Matching):
     xếp hạng ngữ nghĩa trên tập đã lọc, đọc index Chroma đã build sẵn.
     MỚI: nhận thêm extra_signals — color/brand đang ở chế độ ưu tiên mềm,
     dùng để XẾP HẠNG (không loại bỏ) sản phẩm.
  6. few_shot_examples.py + Qwen API -> ResponseService: sinh câu trả lời
     tự nhiên, tuỳ theo action/slot mà DialogueService quyết định.

CHẠY THỬ:
  pip install fastapi uvicorn python-dotenv openai sqlalchemy pymysql sentence-transformers chromadb
  python build_vector_index.py     # 1 lần, index sản phẩm cho AI Matching
  python build_category_index.py   # 1 lần, index tên category cho category_mapper
  uvicorn main:app --reload
  Mở http://127.0.0.1:8000/docs để test qua Swagger UI.

QWEN: đọc key/model từ .env (QWEN_API_KEY/QWEN_BASE_URL/QWEN_MODEL) — hiện
đang trỏ OpenRouter (free/trả phí rẻ). Khi có key Alibaba thật, CHỈ cần
sửa .env, KHÔNG cần sửa file này (need_extractor.py cũng đọc chung 3 biến
này, cũng không cần sửa gì).

MYSQL: đọc DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME từ .env — đã setup
xong theo import_to_mysql.py (bảng `products`, khóa chính `product_id`).
"""

import os
import json
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.session_manager import SessionManager, NextAction
from src.few_shot_examples import build_messages
from src.need_extractor import extract_slots
from src.vector_search import semantic_rank
from src.category_mapper import resolve_leaf_categories
from src.color_mapper import resolve_color
from src.currency import usd_to_vnd, vnd_to_usd, round_vnd, USD_TO_VND_RATE
from src.size_normalization import parse_size_request

# MySQL (optional import — API vẫn chạy được nếu chưa cài, chỉ rule_based_filter trả rỗng)
try:
    from sqlalchemy import create_engine, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

# Qwen qua OpenAI-compatible SDK (OpenRouter hiện tại, đổi sang Alibaba chỉ qua .env)
try:
    from openai import OpenAI
    QWEN_API_KEY = os.getenv("QWEN_API_KEY")
    QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://openrouter.ai/api/v1")
    QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen/qwen-turbo")
    qwen_client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL) if QWEN_API_KEY else None
except ImportError:
    qwen_client = None

app = FastAPI(title="He thong tu van san pham thong minh - API")
session_manager = SessionManager()

SYSTEM_PROMPT = (
    "Bạn là trợ lý tư vấn sản phẩm gia dụng/trang trí nhà cửa (Home & Kitchen). "
    "Chỉ tư vấn dựa trên sản phẩm THẬT có trong hệ thống, không tự bịa sản phẩm. "
    "Trả lời ngắn gọn, tự nhiên, tiếng Việt."
)

DB_URL = (
    f"mysql+pymysql://{os.getenv('DB_USER','root')}:{os.getenv('DB_PASSWORD','')}"
    f"@{os.getenv('DB_HOST','localhost')}:{os.getenv('DB_PORT','3306')}/{os.getenv('DB_NAME','klcn_chatbot')}"
    f"?charset=utf8mb4"
)
PRODUCTS_TABLE = "products"

# Dung sai khi lọc kích thước sản phẩm (product_size_text) — % sai lệch cho phép mỗi
# chiều. 0.20 là giá trị KHỞI ĐẦU lấy theo ví dụ trong tài liệu bàn giao v4 (mục 6/7),
# CHƯA được tối ưu theo từng loại sản phẩm (tài liệu tự ghi rõ "cần chính sách theo
# loại sản phẩm"). Nhóm nên tinh chỉnh giá trị này sau khi thử nghiệm thật.
PRODUCT_SIZE_TOLERANCE = 0.20

# MỚI — tính năng "gợi ý sản phẩm gần nhất". Các field CÓ THỂ nới lỏng khi tìm
# kiếm chặt chẽ ra 0 kết quả (không bao giờ nới category — đổi category là đổi
# hẳn loại sản phẩm, không còn là "gần" nữa; size_space không nằm trong danh
# sách vì nó vốn không được dùng để lọc, xem session_manager.py).
RELAXABLE_FIELDS = ["price_range", "product_size_text", "color", "brand"]
MAX_NEAR_RESULTS = 5


# ---------- Schemas ----------
class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    action: str
    slots: Dict[str, Any]
    recommended_products: List[Dict[str, Any]]


# ---------- Mô-đun 2: Rule-based lọc sản phẩm (MySQL) ----------
def rule_based_filter(
    slots: Dict[str, Any],
    hard_constraint_slots: Optional[Set[str]] = None,
    limit: int = 30,
) -> List[Dict[str, Any]]:
    """Lọc sản phẩm ứng viên theo các slot đã thu thập được.

    hard_constraint_slots: subset của {"color", "brand"} — các field mà user
    đã nói RÕ là bắt buộc (xem session_manager.update_hard_constraints()).
    None/rỗng nghĩa là color/brand hiện có (nếu có) đều chỉ là ƯU TIÊN MỀM,
    KHÔNG lọc cứng SQL — đúng Bảng 2.9 khóa luận.

    Đã sửa (lỗi lệch ngôn ngữ VI/EN đã phát hiện):
      - category: dùng category_mapper.resolve_leaf_categories() (vector
        similarity) thay vì LIKE trực tiếp câu tiếng Việt -> leaf_category
        IN (...). ĐÃ BỎ lọc theo main_category (kiểm tra thật: chỉ có đúng
        1 giá trị "Home_and_Kitchen" cho toàn catalog, không có tác dụng).
      - color: dùng color_mapper.resolve_color() để dịch màu tiếng Việt
        sang tiếng Anh trước khi so khớp.

    MỚI (hard/soft cho color, brand):
      - Nếu color/brand có giá trị NHƯNG KHÔNG nằm trong hard_constraint_slots
        -> KHÔNG lọc cứng SQL (để tầng 2 vector_search dùng làm ưu tiên mềm).
      - Nếu color/brand nằm trong hard_constraint_slots (user nói bắt buộc)
        NHƯNG không dịch/khớp được (vd color không có trong color_mapper)
        -> trả về [] NGAY (0 candidates), KHÔNG bỏ qua điều kiện — đúng
        nguyên tắc "không tự nới ràng buộc bắt buộc" (mục 3.4.1 khóa luận).
        Đây là hành vi CHỦ ĐÍCH, không phải bug — khác với category (luôn
        bắt buộc theo cấu trúc, không có khái niệm "ưu tiên mềm" để category
        rơi vào, nên category không dịch được thì fallback bỏ lọc, xem
        category_mapper.py để biết lý do khác biệt này).

    Trả về [] nếu chưa cài sqlalchemy/pymysql hoặc MySQL chưa sẵn sàng.
    """
    hard_constraint_slots = hard_constraint_slots or set()

    if not SQLALCHEMY_AVAILABLE:
        return []

    try:
        engine = create_engine(DB_URL)
    except Exception as e:
        print(f"[rule_based_filter] Không tạo được engine MySQL: {e}")
        return []

    conditions = []
    params: Dict[str, Any] = {}

    category_text = slots.get("category")
    if category_text:
        matched_categories = resolve_leaf_categories(category_text)
        if matched_categories:
            placeholders = []
            for i, cat in enumerate(matched_categories):
                key = f"cat{i}"
                placeholders.append(f":{key}")
                params[key] = cat
            conditions.append(f"leaf_category IN ({', '.join(placeholders)})")
        else:
            print(f"[rule_based_filter] category='{category_text}' không map được "
                  f"leaf_category nào -> bỏ qua lọc cứng category ở lượt này.")

    color_text = slots.get("color")
    if color_text and "color" in hard_constraint_slots:
        color_en = resolve_color(color_text)
        if color_en:
            # Dataset v3: theo đúng chỉ dẫn của người xử lý dữ liệu màu
            # (xem DOC_TRUOC.md, mục "Cập nhật chatbot của nhóm") — KHÔNG
            # dùng phép bằng hay LIKE trên cột `color` (chuỗi ghép nhiều
            # màu bằng " | ", không đáng tin để so khớp chuỗi con). Dùng
            # JSON_CONTAINS trên colors_json (mảng JSON các màu chuẩn) để
            # lọc CHÍNH XÁC sản phẩm có màu này là 1 THÀNH VIÊN trong mảng,
            # đúng với cả sản phẩm 1 màu lẫn nhiều màu.
            # Yêu cầu: MySQL hỗ trợ JSON_CONTAINS (MySQL >= 5.7.8) và
            # colors_json phải là chuỗi JSON hợp lệ (kiểm tra bằng
            # JSON_VALID sau khi import — xem DOC_TRUOC.md mục 3).
            conditions.append("JSON_CONTAINS(colors_json, :color_json) = 1")
            params["color_json"] = json.dumps(color_en, ensure_ascii=False)
        else:
            # Bắt buộc nhưng không dịch được -> trả 0 NGAY, không âm thầm
            # bỏ ràng buộc (khác hẳn nhánh category ở trên, xem docstring).
            print(f"[rule_based_filter] color='{color_text}' được yêu cầu BẮT BUỘC "
                  f"nhưng không nhận diện được -> trả 0 sản phẩm (không tự nới "
                  f"ràng buộc bắt buộc).")
            return []
    # color có giá trị nhưng KHÔNG bắt buộc -> cố tình KHÔNG thêm điều kiện
    # SQL ở đây, để main.py truyền color_text vào extra_signals cho tầng 2.

    brand_text = slots.get("brand")
    if brand_text and "brand" in hard_constraint_slots:
        conditions.append("brand LIKE :brand")
        params["brand"] = f"%{brand_text}%"
    # brand có giá trị nhưng KHÔNG bắt buộc -> tương tự color, để tầng 2 xử lý.

    # MỚI (dataset v4) — lọc kích thước SẢN PHẨM (khác size_space là kích thước
    # KHÔNG GIAN/phòng, xem need_extractor.py để phân biệt 2 khái niệm này).
    # Luôn lọc CỨNG khi parse được (không có cơ chế hard_constraints riêng như
    # color/brand) — vì sai kích thước khiến sản phẩm không dùng được, hậu quả
    # nặng hơn hẳn lệch màu/thương hiệu, nên không cần user phải nói "bắt buộc"
    # mới lọc cứng. Đặt TRƯỚC khi build LIMIT — đúng yêu cầu tài liệu bàn giao
    # v4 mục 7: "Không nhét bước này sau khi lấy 30 ứng viên".
    size_text = slots.get("product_size_text")
    if size_text:
        size_pair = parse_size_request(size_text)
        if size_pair:
            dim1, dim2 = size_pair  # đã sắp giảm dần: dim1 >= dim2
            tol = PRODUCT_SIZE_TOLERANCE
            conditions.append("size_status = 'recognized'")
            conditions.append("size_confidence IN ('high', 'medium')")
            conditions.append("size_dim1_cm BETWEEN :size_d1_lo AND :size_d1_hi")
            conditions.append("size_dim2_cm BETWEEN :size_d2_lo AND :size_d2_hi")
            params["size_d1_lo"] = dim1 * (1 - tol)
            params["size_d1_hi"] = dim1 * (1 + tol)
            params["size_d2_lo"] = dim2 * (1 - tol)
            params["size_d2_hi"] = dim2 * (1 + tol)
        else:
            # Không parse được (vd "cỡ vừa vừa", "trung bình") -> bỏ qua lọc
            # kích thước ở lượt này, KHÔNG suy đoán con số — đúng nguyên tắc
            # đã áp dụng cho category/color khi không dịch được.
            print(f"[rule_based_filter] product_size_text='{size_text}' không parse được "
                  f"kích thước rõ ràng -> bỏ qua lọc cứng kích thước ở lượt này.")

    price_range = slots.get("price_range")
    if price_range and isinstance(price_range, str) and "-" in price_range:
        try:
            lo_vnd, hi_vnd = price_range.split("-")
            # QUAN TRỌNG: price_range do need_extractor.py trích xuất luôn
            # tính theo VND (Qwen hiểu "dưới 500k" -> "0-500000" theo thói
            # quen người Việt), NHƯNG cột `price` trong DB là USD (dữ liệu
            # gốc Amazon, xem KET_QUA_V3.md: "Dữ liệu giá vẫn là USD.").
            # PHẢI quy đổi VND -> USD (tỷ giá cố định, xem currency.py)
            # trước khi đưa vào so sánh, nếu không sẽ lọc SAI HOÀN TOÀN
            # (vd lọc "price BETWEEN 0 AND 500000" USD sẽ ra hầu như mọi
            # sản phẩm, vì 500.000 USD là con số vô lý cho đồ gia dụng).
            price_lo_usd = vnd_to_usd(float(lo_vnd))
            price_hi_usd = vnd_to_usd(float(hi_vnd))
            conditions.append("price BETWEEN :price_lo AND :price_hi")
            params["price_lo"] = price_lo_usd
            params["price_hi"] = price_hi_usd
        except ValueError:
            pass  # price_range không đúng định dạng "min-max" -> bỏ qua, không lọc giá

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    query = f"SELECT * FROM {PRODUCTS_TABLE} {where_clause} LIMIT :limit"
    params["limit"] = limit

    try:
        with engine.connect() as conn:
            rows = conn.execute(text(query), params).mappings().all()
            return [dict(r) for r in rows]
    except Exception as e:
        print(f"[rule_based_filter] Lỗi truy vấn MySQL: {e}")
        return []


def build_soft_signals(slots: Dict[str, Any], hard_constraint_slots: Set[str]) -> List[str]:
    """Gom các giá trị color/brand hiện có NHƯNG KHÔNG nằm trong
    hard_constraint_slots (tức không bị lọc cứng ở rule_based_filter) —
    dùng làm tín hiệu ưu tiên mềm truyền vào vector_search.semantic_rank(),
    để chúng vẫn ảnh hưởng tới XẾP HẠNG dù không LOẠI sản phẩm thiếu dữ liệu."""
    signals = []
    color_text = slots.get("color")
    if color_text and "color" not in hard_constraint_slots:
        signals.append(color_text)
    brand_text = slots.get("brand")
    if brand_text and "brand" not in hard_constraint_slots:
        signals.append(brand_text)
    return signals


def describe_mismatch(field: str, slots: Dict[str, Any], product: Dict[str, Any]) -> str:
    """Viết 1 câu NGẮN, CỤ THỂ mô tả sản phẩm `product` lệch tiêu chí `field` ra
    sao so với yêu cầu user đã nêu trong `slots` — dùng để đưa vào chỉ dẫn cho
    Qwen, để Qwen nói THẬT với user (không ngụ ý sản phẩm đã khớp đủ)."""
    if field == "price_range":
        try:
            price_vnd = round_vnd(usd_to_vnd(float(product.get("price"))))
            _, hi_vnd = slots["price_range"].split("-")
            return f"giá khoảng {price_vnd:,.0f}đ, vượt ngân sách {float(hi_vnd):,.0f}đ bạn đã nêu"
        except (TypeError, ValueError, KeyError):
            return "giá vượt ngân sách bạn đã nêu"
    if field == "product_size_text":
        d1, d2 = product.get("size_dim1_cm"), product.get("size_dim2_cm")
        if d1 and d2:
            return f"kích thước khoảng {d1:.0f}x{d2:.0f}cm, không đúng cỡ \"{slots['product_size_text']}\" bạn muốn"
        return f"chưa rõ kích thước, không xác nhận được có đúng cỡ \"{slots['product_size_text']}\" bạn muốn không"
    if field == "color":
        actual = product.get("color") or "chưa rõ màu"
        return f"màu {actual}, không đúng màu \"{slots.get('color')}\" bạn yêu cầu"
    if field == "brand":
        actual = product.get("brand") or "chưa rõ thương hiệu"
        return f"thương hiệu {actual}, không đúng thương hiệu \"{slots.get('brand')}\" bạn yêu cầu"
    return "có 1 tiêu chí chưa khớp hoàn toàn"


def find_near_matches(
    slots: Dict[str, Any],
    hard_constraint_slots: Set[str],
    session_id: str,
) -> List[Dict[str, Any]]:
    """Khi tìm kiếm CHẶT CHẼ (đủ mọi điều kiện) ra 0 kết quả, thử nới LẦN LƯỢT
    TỪNG tiêu chí một trong RELAXABLE_FIELDS (không bao giờ nới category, và
    không bao giờ nới quá 1 tiêu chí cùng lúc — nới nhiều tiêu chí một lần sẽ
    không còn biết chính xác sản phẩm lệch ở đâu để nói thật với user).

    Mỗi sản phẩm tìm được sẽ gắn `match_type: "near"` và `near_reason` (mô tả
    CỤ THỂ lệch gì) — KHÔNG bao giờ trộn lẫn với sản phẩm khớp đủ (những sản
    phẩm khớp đủ thì rule_based_filter() gốc đã trả về rồi, hàm này CHỈ được
    gọi khi kết quả gốc là rỗng)."""
    results: List[Dict[str, Any]] = []
    seen_ids = set()

    for field in RELAXABLE_FIELDS:
        # color/brand chỉ đáng nới nếu ĐANG là ràng buộc cứng thật sự — nếu nó
        # đã là ưu tiên mềm từ đầu thì rule_based_filter() gốc đã không lọc
        # cứng theo nó rồi, "nới" nó ở đây sẽ không tạo ra khác biệt gì.
        if field in ("color", "brand"):
            if not slots.get(field) or field not in hard_constraint_slots:
                continue
        else:
            if not slots.get(field):
                continue

        relaxed_slots = dict(slots)
        relaxed_hard = set(hard_constraint_slots)
        if field in ("price_range", "product_size_text"):
            relaxed_slots[field] = None
        else:
            relaxed_hard.discard(field)

        candidates = rule_based_filter(relaxed_slots, relaxed_hard)
        candidates = session_manager.filter_out_already_shown(session_id, candidates)
        # Vẫn xếp hạng theo `slots` GỐC (chưa nới) — để các tiêu chí ngữ nghĩa
        # khác (style/material/free_text/soft color-brand) vẫn được tôn trọng
        # khi chọn sản phẩm nào "gần nhất" trong tập đã nới.
        ranked = semantic_rank(candidates, slots, top_k=3, extra_signals=build_soft_signals(slots, hard_constraint_slots))

        for p in ranked:
            pid = p.get("product_id")
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            p["match_type"] = "near"
            p["near_reason"] = describe_mismatch(field, slots, p)
            results.append(p)

    results.sort(key=lambda p: p.get("_similarity_score", 0), reverse=True)
    return results[:MAX_NEAR_RESULTS]


def get_price_stats(category: Optional[str]) -> Optional[Dict[str, float]]:
    """Truy vấn giá MIN/MAX/trung bình (USD, đúng theo dữ liệu gốc trong DB)
    của các sản phẩm cùng category — dùng để GỢI Ý khoảng giá tham khảo cho
    user khi hỏi price_range, thay vì bắt user tự đoán mù giá. Dùng
    category_mapper để dịch category tiếng Việt sang leaf_category tiếng
    Anh thật trước khi query. Trả về None nếu chưa có category, không map
    được leaf_category nào, hoặc lỗi DB.

    Trả về CẢ 2 đơn vị: *_usd (giá trị THẬT, đúng dữ liệu gốc) và *_vnd
    (quy đổi theo tỷ giá CỐ ĐỊNH ở currency.py, chỉ mang tính THAM KHẢO —
    xem giải thích trong currency.py) — vì user Việt Nam nói/hiểu giá theo
    VND, nhưng dữ liệu gốc là USD (xem rule_based_filter() để biết vì sao)."""
    if not category or not SQLALCHEMY_AVAILABLE:
        return None

    matched_categories = resolve_leaf_categories(category)
    if not matched_categories:
        return None

    try:
        engine = create_engine(DB_URL)
        placeholders = [f":cat{i}" for i in range(len(matched_categories))]
        params = {f"cat{i}": cat for i, cat in enumerate(matched_categories)}
        params["zero"] = 0
        query = text(
            f"SELECT MIN(price) AS min_p, MAX(price) AS max_p, AVG(price) AS avg_p "
            f"FROM {PRODUCTS_TABLE} "
            f"WHERE leaf_category IN ({', '.join(placeholders)}) AND price > :zero"
        )
        with engine.connect() as conn:
            row = conn.execute(query, params).mappings().first()
        if not row or row["min_p"] is None:
            return None
        min_usd, max_usd, avg_usd = float(row["min_p"]), float(row["max_p"]), float(row["avg_p"])
        return {
            "min_usd": min_usd, "max_usd": max_usd, "avg_usd": avg_usd,
            "min_vnd": usd_to_vnd(min_usd), "max_vnd": usd_to_vnd(max_usd), "avg_vnd": usd_to_vnd(avg_usd),
        }
    except Exception as e:
        print(f"[get_price_stats] Lỗi truy vấn: {e}")
        return None


# ---------- Mô-đun 4: Sinh phản hồi bằng Qwen, tuỳ theo action/slot ----------
def build_action_instruction(
    action: NextAction, slot: Optional[str], top_products: List[Dict[str, Any]], slots: Dict[str, Any]
) -> str:
    """Tạo câu chỉ dẫn ngắn gắn kèm câu hỏi user, báo cho Qwen biết PHẢI làm
    gì ở lượt này — Qwen sẽ tự viết câu chữ tự nhiên theo đúng phong cách
    few-shot, không lặp lại máy móc."""
    if action == NextAction.ASK_HARD_SLOT and slot == "category":
        return "[Chỉ dẫn hệ thống: hãy hỏi user đang cần tìm loại sản phẩm gì.]"
    if action == NextAction.ASK_HARD_SLOT and slot == "size_space":
        return ("[Chỉ dẫn hệ thống: hãy hỏi user về kích thước/không gian sử dụng "
                "(không ép phải có số đo chính xác, chấp nhận mô tả mơ hồ).]")
    if action == NextAction.ASK_SOFT_SLOT and slot == "price_range":
        stats = get_price_stats(slots.get("category"))
        if stats:
            return (
                f"[Chỉ dẫn hệ thống: hãy hỏi user về ngân sách/khoảng giá mong muốn (tính bằng "
                f"VND, vì user là người Việt). Sản phẩm loại này trong hệ thống dao động khoảng "
                f"từ {round_vnd(stats['min_vnd']):,.0f}đ đến {round_vnd(stats['max_vnd']):,.0f}đ "
                f"(trung bình khoảng {round_vnd(stats['avg_vnd']):,.0f}đ, quy đổi theo tỷ giá tham "
                f"khảo 1 USD = {USD_TO_VND_RATE:,}đ, không phải giá niêm yết chính thức) — hãy nêu "
                f"khoảng giá này để gợi ý cho user tham khảo, vì user có thể chưa biết mức giá hợp "
                f"lý của loại sản phẩm này.]"
            )
        return "[Chỉ dẫn hệ thống: hãy hỏi user về ngân sách/khoảng giá mong muốn (tính bằng VND).]"
    if action == NextAction.ASK_SOFT_SLOT and slot:
        return f"[Chỉ dẫn hệ thống: hãy hỏi user về thuộc tính '{slot}' (màu/chất liệu/phong cách/thương hiệu).]"
    if action == NextAction.SEARCH_PRODUCTS:
        if not top_products:
            return ("[Chỉ dẫn hệ thống: KHÔNG tìm thấy sản phẩm nào khớp đủ điều kiện, kể cả "
                     "gần đúng. Hãy báo rõ cho user và đề nghị điều chỉnh tiêu chí (nới ngân "
                     "sách, đổi màu, bỏ bớt ràng buộc...). KHÔNG nói sẽ 'kiểm tra thêm' hay "
                     "'chờ một chút' — đây đã là kết quả cuối cùng của lượt tìm kiếm này, không "
                     "có xử lý nào chạy ngầm phía sau.]")

        near = [p for p in top_products if p.get("match_type") == "near"]
        if near:
            # TOÀN BỘ kết quả đều là "gần nhất" (rule_based_filter gốc ra 0,
            # find_near_matches mới tìm được) — PHẢI nói RÕ đây KHÔNG PHẢI
            # khớp hoàn toàn, nêu cụ thể từng mẫu lệch điểm gì, để user tự
            # quyết định có chấp nhận không, KHÔNG được ngụ ý đã đáp ứng đủ.
            lines = "; ".join(
                f"{p.get('title', '?')} (lệch: {p.get('near_reason', 'chưa rõ')})"
                for p in near[:5]
            )
            return (
                f"[Chỉ dẫn hệ thống: KHÔNG có sản phẩm nào khớp HOÀN TOÀN yêu cầu user đã nêu. "
                f"Đây là các sản phẩm GẦN NHẤT tìm được, mỗi mẫu CHỈ lệch đúng 1 tiêu chí (đã "
                f"ghi rõ lệch gì ngay sau tên): {lines}. Hãy nói RÕ RÀNG ngay từ đầu đây là gợi "
                f"ý gần nhất chứ KHÔNG PHẢI khớp hoàn toàn, sau đó nêu CỤ THỂ từng mẫu lệch điểm "
                f"gì (dùng đúng thông tin lệch đã cho, không tự suy thêm), để user tự quyết định "
                f"có chấp nhận mẫu nào không hay muốn đổi tiêu chí. TUYỆT ĐỐI KHÔNG khẳng định "
                f"các mẫu này đáp ứng đủ yêu cầu ban đầu.]"
            )

        names = ", ".join(p.get("title", "?") for p in top_products[:5])
        return (
            f"[Chỉ dẫn hệ thống: đây là TOÀN BỘ kết quả tìm được ở lượt này (không có thêm, "
            f"không có xử lý nào đang chạy ngầm phía sau) — hãy giới thiệu các sản phẩm sau cho "
            f"user: {names}. Nếu bạn thấy có điểm nào trong tên/mô tả sản phẩm có vẻ CHƯA khớp "
            f"hoàn toàn với yêu cầu user đã nêu (vd kích thước, chất liệu), hãy NÊU RÕ điều đó để "
            f"user tự cân nhắc có phù hợp không, thay vì lặng lẽ giới thiệu như đã khớp hoàn hảo. "
            f"KHÔNG nói sẽ 'kiểm tra thêm' hay 'chờ một chút' — không có bước xử lý tiếp theo nào "
            f"sẽ tự động chạy cho tới khi user nhắn tin mới.]"
        )
    if action == NextAction.END_SESSION:
        return "[Chỉ dẫn hệ thống: user muốn kết thúc/đã chốt mua, hãy chào tạm biệt lịch sự.]"
    return ""


def generate_llm_reply(
    session_id: str, user_message: str, action: NextAction, slot: Optional[str],
    top_products: List[Dict[str, Any]], slots: Dict[str, Any]
) -> str:
    if qwen_client is None:
        return "[Chưa cấu hình QWEN_API_KEY trong .env nên chưa sinh được phản hồi tự nhiên.]"

    history = session_manager.get_history(session_id)
    instruction = build_action_instruction(action, slot, top_products, slots)
    augmented_message = f"{user_message}\n{instruction}" if instruction else user_message

    messages = build_messages(
        system_prompt=SYSTEM_PROMPT,
        user_new_question=augmented_message,
        conversation_history=history[:-1] if history else None,  # bỏ tin nhắn cuối, đã đưa vào augmented_message
    )

    try:
        response = qwen_client.chat.completions.create(
            model=QWEN_MODEL,
            messages=messages,
            max_tokens=600,  # tăng từ 400 -> 600, chừa chỗ nếu provider tính cả token "suy nghĩ" vào đây
            extra_headers={  # OpenRouter khuyến nghị — vô hại nếu đổi sang Alibaba
                "HTTP-Referer": "https://github.com/",
                "X-Title": "KLCN Chatbot Home&Kitchen",
            },
            extra_body={"reasoning": {"enabled": False}},  # tắt "thinking" — tránh model tốn hết
            # max_tokens cho suy luận nội bộ mà chưa kịp viết câu trả lời thật (content = None)
        )
        content = response.choices[0].message.content
        if not content:
            print("[generate_llm_reply] Model trả về content rỗng/None — dùng câu trả lời dự phòng.")
            return ("Xin lỗi, hệ thống tư vấn đang bận xử lý, bạn vui lòng nhắn lại câu hỏi "
                    "hoặc thử lại sau giây lát nhé.")
        return content
    except Exception as e:
        return f"[Lỗi gọi Qwen API: {e}]"


# ---------- Endpoint chính ----------
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message không được để trống")

    session = session_manager.get_or_create(req.session_id)
    session_manager.add_message(session.session_id, "user", req.message)

    # NeedExtractor: trích xuất slot mới từ tin nhắn (1 lệnh gọi Qwen riêng,
    # tách khỏi ResponseService — đúng Bảng 3.1 khóa luận).
    current_slots = session_manager.get_slots(session.session_id)
    new_slots = extract_slots(req.message, current_slots, session_manager.get_history(session.session_id))

    # MỚI: tách "hard_constraints" ra riêng TRƯỚC khi gọi update_slots() —
    # update_slots() sẽ tự bỏ qua key này vì không nằm trong ALL_SLOTS
    # (xem session_manager.py), nên phải xử lý riêng ở đây.
    hard_constraints_this_turn = new_slots.pop("hard_constraints", [])
    if hard_constraints_this_turn:
        session_manager.update_hard_constraints(session.session_id, hard_constraints_this_turn)
    if new_slots:
        session_manager.update_slots(session.session_id, new_slots)

    decision = session_manager.decide_next_action(session.session_id, req.message)
    action, slot = decision["action"], decision["slot"]

    if action == NextAction.ASK_HARD_SLOT and slot == "size_space":
        session_manager.mark_size_space_asked(session.session_id)

    top_products: List[Dict[str, Any]] = []
    if action == NextAction.SEARCH_PRODUCTS:
        slots = session_manager.get_slots(session.session_id)
        hard_constraints = session_manager.get_hard_constraints(session.session_id)

        candidates = rule_based_filter(slots, hard_constraints)
        candidates = session_manager.filter_out_already_shown(session.session_id, candidates)

        soft_signals = build_soft_signals(slots, hard_constraints)
        top_products = semantic_rank(candidates, slots, extra_signals=soft_signals)

        if top_products:
            for p in top_products:
                p["match_type"] = "exact"
        else:
            # MỚI — tìm kiếm chặt chẽ ra 0 kết quả: thử tìm sản phẩm GẦN NHẤT
            # (nới từng tiêu chí một, không bao giờ nới category). Nếu vẫn
            # không có gì, top_products giữ nguyên [] -> build_action_instruction()
            # sẽ tự báo "không có, kể cả gần đúng" (nhánh đã có sẵn).
            top_products = find_near_matches(slots, hard_constraints, session.session_id)

        # Gắn thêm giá VND ƯỚC TÍNH (quy đổi theo tỷ giá cố định, xem
        # currency.py) vào từng sản phẩm trả về — KHÔNG thay thế trường
        # `price` gốc (vẫn giữ nguyên USD, đúng dữ liệu thật trong DB, để
        # kiểm chứng/đối chiếu được). Đặt tên "_estimate" để minh bạch đây
        # là số ước tính, không phải giá niêm yết chính thức.
        for p in top_products:
            raw_price = p.get("price")
            if raw_price is not None:
                try:
                    p["price_vnd_estimate"] = round_vnd(usd_to_vnd(float(raw_price)))
                except (TypeError, ValueError):
                    pass  # giá không parse được thành số -> bỏ qua, không bịa

        session_manager.mark_as_shown(session.session_id, top_products)

    reply = generate_llm_reply(session.session_id, req.message, action, slot, top_products, session_manager.get_slots(session.session_id))
    session_manager.add_message(session.session_id, "assistant", reply)

    return ChatResponse(
        session_id=session.session_id,
        reply=reply,
        action=action.value,
        slots=session_manager.get_slots(session.session_id),
        recommended_products=top_products,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mysql_lib_available": SQLALCHEMY_AVAILABLE,
        "qwen_configured": qwen_client is not None,
        "qwen_model": QWEN_MODEL if qwen_client else None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)