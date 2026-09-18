"""
need_extractor.py
Mô-đun NeedExtractor (đúng theo Bảng 3.1 — Trách nhiệm và kết quả trao đổi
giữa các dịch vụ, Chương 3 khóa luận):
  Đầu vào: Tin nhắn và ngữ cảnh cần thiết
  Đầu ra:  Các thao tác thêm, sửa, bỏ tiêu chí; điểm chưa rõ cần xác nhận

Cách làm: gọi Qwen với system prompt YÊU CẦU CHỈ trả JSON (không kèm chữ
nào khác), theo đúng schema slot của session_manager.ALL_SLOTS. Đây là 1
lệnh gọi Qwen RIÊNG, tách khỏi generate_llm_reply() trong main.py — đúng
tinh thần "tách vai trò" của khóa luận (NeedExtractor khác ResponseService).

Lưu ý chi phí: mỗi lượt chat sẽ tốn 2 lần gọi Qwen (1 lần extract slot ở
đây, 1 lần sinh câu trả lời ở main.py). Với model rẻ (qwen-turbo) và quy
mô demo khóa luận, chi phí này vẫn không đáng kể.

Nếu JSON trả về không hợp lệ (model trả lẫn chữ, sai format), hàm sẽ trả
về {} (không cập nhật gì) và IN CẢNH BÁO ra console — KHÔNG được tự bịa
giá trị mặc định, đúng nguyên tắc "không ghi đè trạng thái nhu cầu bằng
dữ liệu lỗi" đã ghi trong đặc tả UC-07 (mục 2.4.1 khóa luận).
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://openrouter.ai/api/v1")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen/qwen-turbo")

_client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL) if (OpenAI and QWEN_API_KEY) else None

# Khớp đúng ALL_SLOTS trong session_manager.py — sửa cả 2 nơi nếu đổi slot.
SLOT_SCHEMA_HINT = (
    '{"category": string|null, "price_range": "min-max"|null, '
    '"color": string|null, "material": string|null, "style": string|null, '
    '"brand": string|null, "size_space": string|null}'
)

EXTRACT_SYSTEM_PROMPT = (
    "Bạn là bộ trích xuất thông tin cho hệ thống tư vấn sản phẩm gia dụng. "
    "CHỈ trả về đúng 1 object JSON, KHÔNG kèm chữ giải thích, KHÔNG dùng markdown code block. "
    f"Schema bắt buộc: {SLOT_SCHEMA_HINT}. "
    "Chỉ điền trường nào user THỰC SỰ nhắc tới trong tin nhắn MỚI NHẤT; trường không nhắc tới "
    "để null (không suy diễn, không tự đặt giá trị mặc định). "
    "price_range chỉ điền khi user nêu rõ số tiền/khoảng giá, viết dạng 'min-max' bằng số "
    "(ví dụ user nói 'dưới 500k' -> '0-500000'; 'khoảng 200-300k' -> '200000-300000')."
)


def _build_context_note(current_slots: Dict[str, Any]) -> str:
    known = {k: v for k, v in current_slots.items() if v and k != "other_notes"}
    if not known:
        return "Nhu cầu đã biết trước đó: (chưa có gì)."
    return f"Nhu cầu đã biết trước đó (để tham chiếu, KHÔNG lặp lại nếu tin nhắn mới không nhắc tới): {known}"


def extract_slots(
    user_message: str,
    current_slots: Optional[Dict[str, Any]] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Trả về dict các slot MỚI trích xuất được từ user_message (chỉ chứa
    trường có giá trị thật, không chứa trường null) — dùng trực tiếp cho
    session_manager.update_slots().
    Trả về {} nếu chưa cấu hình Qwen hoặc parse JSON thất bại."""
    if _client is None:
        print("[need_extractor] Chưa cấu hình QWEN_API_KEY — bỏ qua trích xuất.")
        return {}

    current_slots = current_slots or {}
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
        {"role": "user", "content": f"{_build_context_note(current_slots)}\n\nTin nhắn mới của user: \"{user_message}\""},
    ]

    try:
        response = _client.chat.completions.create(
            model=QWEN_MODEL,
            messages=messages,
            max_tokens=200,
            temperature=0,  # cần ổn định/xác định cho tác vụ trích xuất, không cần sáng tạo
            extra_headers={"HTTP-Referer": "https://github.com/", "X-Title": "KLCN Chatbot Home&Kitchen"},
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[need_extractor] Lỗi gọi Qwen: {e}")
        return {}

    parsed = _safe_parse_json(raw)
    if parsed is None:
        print(f"[need_extractor] Không parse được JSON từ model, bỏ qua lượt này. Raw: {raw[:200]}")
        return {}

    # Chỉ giữ trường có giá trị thật (không None/""), tránh ghi đè slot cũ bằng null.
    return {k: v for k, v in parsed.items() if v not in (None, "", "null")}


def _safe_parse_json(raw: str) -> Optional[Dict[str, Any]]:
    """Cố gắng parse JSON kể cả khi model lỡ kèm ```json ... ``` hoặc vài chữ thừa."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None


if __name__ == "__main__":
    # Demo nhanh — cần QWEN_API_KEY trong .env để chạy thật.
    demo_slots = {"category": "thảm trải sàn"}
    result = extract_slots("Khoảng 2m x 3m, màu be, dưới 500k nhé", demo_slots)
    print("Slot trích xuất được:", result)