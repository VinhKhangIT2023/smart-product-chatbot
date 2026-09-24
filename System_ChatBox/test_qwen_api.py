"""
test_qwen_api.py
Script test gọi model Qwen qua OpenRouter (free tier, không cần thẻ) —
dùng để kiểm tra API key + kết nối trước khi tích hợp vào main.py
(mô-đun "Sinh phản hồi tư vấn bằng LLM").

Đang dùng OpenRouter thay vì Alibaba Cloud DashScope trực tiếp vì:
  - Free tier của Alibaba yêu cầu xác minh danh tính + có thể bị
    risk-control chặn (RISK.RISK_CONTROL_REJECTION) với tài khoản mới.
  - OpenRouter host thẳng model Qwen thật (cùng trọng số), miễn phí
    hoàn toàn (model có hậu tố ":free"), không cần thẻ, không cần
    xác minh danh tính — phù hợp giai đoạn code/test của khóa luận.
  - Giới hạn free tier OpenRouter: 20 request/phút, 50 request/ngày
    (mỗi request = 1 lần gọi API, không tính theo độ dài nội dung).

Nếu sau này Alibaba xử lý xong risk-control, chỉ cần đổi 3 biến
QWEN_API_KEY/QWEN_BASE_URL/QWEN_MODEL trong .env, KHÔNG cần sửa code.

Chạy:
    pip install openai python-dotenv
    python test_qwen_api.py
"""

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    print("Thiếu thư viện. Chạy: pip install openai")
    sys.exit(1)


# Mặc định trỏ về OpenRouter (free tier). Nếu .env có set khác thì dùng theo .env.
BASE_URL = os.getenv("QWEN_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.getenv("QWEN_API_KEY")
MODEL = os.getenv("QWEN_MODEL", "qwen/qwen3.6-plus:free")


def check_setup() -> bool:
    if not API_KEY:
        print("=" * 60)
        print("CHƯA CÓ QWEN_API_KEY trong file .env — cần setup trước khi chạy file này.")
        print("Xem hướng dẫn lấy key free tại openrouter.ai/keys")
        print("=" * 60)
        return False
    return True


def test_connection():
    if not check_setup():
        return

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Bạn là trợ lý tư vấn sản phẩm gia dụng, trả lời ngắn gọn bằng tiếng Việt."},
                {"role": "user", "content": "Xin chào, bạn có thể giúp tôi tìm thảm trải phòng khách không?"},
            ],
            max_tokens=400,
            # 2 header này OpenRouter khuyến nghị gửi kèm (không bắt buộc để
            # chạy được, nhưng giúp app của bạn hiện đúng tên trên dashboard
            # OpenRouter và tránh bị hạ ưu tiên khi hệ thống đông tải).
            extra_headers={
                "HTTP-Referer": "https://github.com/",
                "X-Title": "KLCN Chatbot Home&Kitchen",
            },
            # Tắt "thinking/reasoning" — model free hay bật mặc định, dễ tốn hết
            # max_tokens cho phần suy luận nội bộ trước khi kịp trả câu trả lời
            # thật (content trả về None) — xem openrouter.ai/docs cho model hỗ trợ.
            extra_body={"reasoning": {"enabled": False}},
        )
        content = response.choices[0].message.content
        if not content:
            print("[CẢNH BÁO] Model trả về content rỗng/None.")
            print("=> Có thể do 'thinking' vẫn tốn hết token dù đã set enabled=False (model không hỗ trợ tắt).")
            print("   Thử tăng max_tokens lên cao hơn, hoặc đổi model khác.")
            return
        print("Kết nối Qwen API (qua OpenRouter) THÀNH CÔNG.")
        print("Model:", MODEL)
        print("Phản hồi mẫu:")
        print(content)
    except Exception as e:
        print("[Lỗi gọi Qwen API]:", e)
        print("=> Kiểm tra lại: API key đúng chưa? Đúng model chưa (phải có hậu tố ':free')?")
        print("   Nếu lỗi 429: có thể đã hết 50 request/ngày, đợi reset sau 24h.")


if __name__ == "__main__":
    test_connection()