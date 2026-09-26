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
from typing import Any, Dict, List, Optional, Set

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
            # Dataset v3: cột `color` có thể chứa NHIỀU giá trị ghép bằng
            # " | " (vd "Black | Gold" cho sản phẩm 2 màu). Dùng LIKE '%X%'
            # đơn giản sẽ khớp NHẦM (vd tìm "Gold" sẽ lọt luôn cả "Rose Gold"
            # dù không có "Gold" đứng riêng). Kỹ thuật bọc dấu phân cách ở cả
            # 2 đầu — CONCAT(' | ', color, ' | ') rồi so khớp '% | X | %' —
            # đảm bảo chỉ khớp đúng NGUYÊN 1 giá trị trong danh sách, dùng
            # được cho cả dòng 1 màu lẫn nhiều màu, không cần đổi schema.
            # (NULL color của các dòng needs_review/missing/non_color tự
            # động bị loại vì CONCAT(..., NULL, ...) = NULL trong MySQL.)
            conditions.append("CONCAT(' | ', color, ' | ') LIKE :color")
            params["color"] = f"% | {color_en} | %"
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


def build_soft_signals(slots: Dict[str, Any], hard_constraint_slots: Set[str]) -> List[str]:
    """MỚI: gom các giá trị color/brand hiện có NHƯNG KHÔNG nằm trong
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


def get_price_stats(category: Optional[str]) -> Optional[Dict[str, float]]:
    """Truy vấn giá MIN/MAX/trung vị của các sản phẩm cùng category — dùng để
    GỢI Ý khoảng giá tham khảo cho user khi hỏi price_range, thay vì bắt user
    tự đoán mù giá. Dùng category_mapper để dịch category tiếng Việt sang
    leaf_category tiếng Anh thật trước khi query (cùng lỗi gốc, cùng cách
    sửa như rule_based_filter()). Trả về None nếu chưa có category, không
    map được leaf_category nào, hoặc lỗi DB."""
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