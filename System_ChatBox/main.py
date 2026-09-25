"""
main.py
File chạy chính — FastAPI backend ghép các mô-đun (đúng Bảng 3.1 khóa luận):
  1. session_manager.py  -> DialogueService: decide_next_action() quyết định
     hành động mỗi lượt: ASK_HARD_SLOT / ASK_SOFT_SLOT / SEARCH_PRODUCTS / END_SESSION
  2. need_extractor.py    -> NeedExtractor: trích xuất/cập nhật slot nhu cầu
     từ tin nhắn user (gọi Qwen riêng, tách khỏi bước sinh câu trả lời)
  3. category_mapper.py / color_mapper.py -> lớp DỊCH thuộc tính tiếng Việt
     (do NeedExtractor trích xuất) sang giá trị tiếng Anh THẬT đang tồn tại
     trong catalog, TRƯỚC KHI đưa vào rule_based_filter(). Đây là phần MỚI
     THÊM để sửa lỗi đã phát hiện: catalog là dữ liệu Amazon tiếng Anh, còn
     few-shot của NeedExtractor dạy trích xuất tiếng Việt -> nếu lọc SQL
     thẳng bằng câu tiếng Việt thì KHÔNG BAO GIỜ khớp -> luôn trả 0 sản phẩm.
  4. rule_based_filter()  -> RecommendationService tầng 1: lọc cứng bằng SQL
     trên bảng `products` (MySQL, đã import 15.714 sản phẩm), SAU KHI đã
     dịch category/color qua category_mapper/color_mapper.
  5. vector_search.py     -> RecommendationService tầng 2 (AI Matching):
     xếp hạng ngữ nghĩa trên tập đã lọc, đọc index Chroma đã build sẵn
     (xem build_vector_index.py — chạy 1 lần offline, KHÔNG chạy mỗi request)
  6. few_shot_examples.py + Qwen API -> ResponseService: sinh câu trả lời
     tự nhiên, tuỳ theo action/slot mà DialogueService quyết định

CHẠY THỬ:
  pip install fastapi uvicorn python-dotenv openai sqlalchemy pymysql sentence-transformers chromadb
  python build_vector_index.py     # 1 lần, index sản phẩm cho AI Matching
  python build_category_index.py   # 1 lần, index tên category cho category_mapper (MỚI)
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
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from session_manager import SessionManager, NextAction
from few_shot_examples import build_messages
from need_extractor import extract_slots
from vector_search import semantic_rank
from category_mapper import resolve_leaf_categories
from color_mapper import resolve_color

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
def rule_based_filter(slots: Dict[str, Any], limit: int = 30) -> List[Dict[str, Any]]:
    """Lọc sản phẩm ứng viên theo các slot đã thu thập được.

    SỬA SO VỚI BẢN CŨ (lỗi 0 candidates đã phát hiện):
      - category: KHÔNG còn LIKE trực tiếp câu tiếng Việt vào main_category/
        leaf_category (catalog là tiếng Anh -> không bao giờ khớp). Thay
        vào đó dùng category_mapper.resolve_leaf_categories() để tìm các
        leaf_category TIẾNG ANH THẬT gần nghĩa nhất bằng vector similarity,
        rồi lọc bằng `leaf_category IN (...)`.
      - ĐÃ BỎ lọc theo main_category: kiểm tra DISTINCT thật cho thấy cột
        này chỉ có đúng 1 giá trị "Home_and_Kitchen" cho toàn bộ catalog —
        lọc theo nó không thu hẹp được gì, chỉ leaf_category mới có tác dụng.
      - color: dùng color_mapper.resolve_color() để dịch màu tiếng Việt
        sang tiếng Anh trước khi LIKE (color là soft slot — không map được
        thì bỏ qua điều kiện, không suy đoán).
      - Nếu category không map được leaf_category nào đủ tin cậy: KHÔNG áp
        lọc cứng category (để AI Matching / free_text xử lý ở tầng 2), có
        log cảnh báo rõ ràng — không tự nới các ràng buộc KHÁC (giá, brand)
        để bù lại.

    Trả về [] nếu chưa cài sqlalchemy/pymysql hoặc MySQL chưa sẵn sàng.
    """
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
            # Không map được category nào đủ tin cậy -> không lọc cứng theo
            # category ở tầng SQL, nhưng vẫn giữ category_text để vector_search.py
            # (build_query_text) có thể dùng làm tín hiệu ngữ nghĩa ở tầng 2.
            print(f"[rule_based_filter] category='{category_text}' không map được "
                  f"leaf_category nào -> bỏ qua lọc cứng category ở lượt này.")

    color_text = slots.get("color")
    if color_text:
        color_en = resolve_color(color_text)
        if color_en:
            conditions.append("color LIKE :color")
            params["color"] = f"%{color_en}%"
        # Không map được màu -> bỏ qua điều kiện màu (soft slot, không suy đoán).

    if slots.get("brand"):
        conditions.append("brand LIKE :brand")
        params["brand"] = f"%{slots['brand']}%"

    price_range = slots.get("price_range")
    if price_range and isinstance(price_range, str) and "-" in price_range:
        try:
            lo, hi = price_range.split("-")
            conditions.append("price BETWEEN :price_lo AND :price_hi")
            params["price_lo"] = float(lo)
            params["price_hi"] = float(hi)
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


# AI Matching (xếp hạng ngữ nghĩa Top-K) đã chuyển sang vector_search.semantic_rank()
# — module riêng, đọc index Chroma đã build sẵn (build_vector_index.py), không còn
# là stub cắt thô top_k như trước.


def get_price_stats(category: Optional[str]) -> Optional[Dict[str, float]]:
    """Truy vấn giá MIN/MAX/trung vị của các sản phẩm cùng category — dùng để
    GỢI Ý khoảng giá tham khảo cho user khi hỏi price_range, thay vì bắt user
    tự đoán mù giá.

    SỬA: dùng category_mapper để dịch category tiếng Việt sang leaf_category
    tiếng Anh thật trước khi query, giống rule_based_filter() (cùng 1 lỗi gốc,
    cùng 1 cách sửa). Trả về None nếu chưa có category, không map được
    leaf_category nào, hoặc lỗi DB — khi đó hệ thống vẫn hỏi giá bình thường,
    chỉ là không kèm gợi ý số liệu."""
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
        return {"min": float(row["min_p"]), "max": float(row["max_p"]), "avg": float(row["avg_p"])}
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
                f"[Chỉ dẫn hệ thống: hãy hỏi user về ngân sách/khoảng giá mong muốn. "
                f"Sản phẩm loại này trong hệ thống dao động từ {stats['min']:,.0f}đ đến "
                f"{stats['max']:,.0f}đ (trung bình khoảng {stats['avg']:,.0f}đ) — hãy nêu "
                f"khoảng giá này để gợi ý cho user tham khảo, vì user có thể chưa biết "
                f"mức giá hợp lý của loại sản phẩm này.]"
            )
        return "[Chỉ dẫn hệ thống: hãy hỏi user về ngân sách/khoảng giá mong muốn.]"
    if action == NextAction.ASK_SOFT_SLOT and slot:
        return f"[Chỉ dẫn hệ thống: hãy hỏi user về thuộc tính '{slot}' (màu/chất liệu/phong cách/thương hiệu).]"
    if action == NextAction.SEARCH_PRODUCTS:
        names = ", ".join(p.get("title", "?") for p in top_products[:5]) or "(chưa có sản phẩm khớp)"
        return f"[Chỉ dẫn hệ thống: đủ thông tin rồi, hãy gợi ý các sản phẩm sau cho user: {names}]"
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
    if new_slots:
        session_manager.update_slots(session.session_id, new_slots)

    decision = session_manager.decide_next_action(session.session_id, req.message)
    action, slot = decision["action"], decision["slot"]

    if action == NextAction.ASK_HARD_SLOT and slot == "size_space":
        session_manager.mark_size_space_asked(session.session_id)

    top_products: List[Dict[str, Any]] = []
    if action == NextAction.SEARCH_PRODUCTS:
        slots = session_manager.get_slots(session.session_id)
        candidates = rule_based_filter(slots)
        candidates = session_manager.filter_out_already_shown(session.session_id, candidates)
        top_products = semantic_rank(candidates, slots)
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