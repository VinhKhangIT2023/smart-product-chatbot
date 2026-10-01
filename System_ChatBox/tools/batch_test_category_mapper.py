"""
batch_test_category_mapper.py
Kiểm tra HÀNG LOẠT category_mapper.resolve_leaf_categories() với các danh từ
tiếng Việt PHỔ BIẾN trong Home & Kitchen — chạy 1 LẦN để phát hiện nhiều lỗi
kiểu "nồi" cùng lúc, thay vì phát hiện rải rác từng cái một qua /chat.

VỊ TRÍ: file này nằm trong tools/, NHƯNG category_mapper.py (file nó cần
import) nằm ở thư mục GỐC System_ChatBox/ (1 cấp cha) — khác thư mục nên
`from category_mapper import ...` trực tiếp SẼ LỖI (Python không tự tìm
lên thư mục cha). Đoạn thêm `sys.path` bên dưới xử lý đúng việc này, để
file vẫn chạy được dù đặt trong tools/ hay gọi từ bất kỳ đâu.

CÁCH ĐỌC KẾT QUẢ (tự bạn xem, không cần gửi hết cho tôi):
  - Nếu danh sách trả về NGHE HỢP LÝ (đúng loại sản phẩm) -> bỏ qua, không
    cần làm gì.
  - Nếu danh sách RỖNG [] -> không phải lỗi (chỉ là model "thận trọng", coi
    như chưa đủ tin cậy) -> KHÔNG cần vá, hệ thống tự bỏ qua lọc category ở
    trường hợp này (rule_based_filter fallback sang AI Matching).
  - CHỈ đáng báo cho tôi nếu danh sách CÓ giá trị NHƯNG chứa toàn category
    KHÔNG LIÊN QUAN (giống ca "nồi" ra "Vacuums") -> đây mới là lỗi cần vá
    bằng VI_CATEGORY_HINTS.

CHẠY (từ BẤT KỲ đâu, không bắt buộc đứng ở System_ChatBox/ nữa nhờ sys.path
bên dưới — nhưng NHỚ: category_mapper.py cần category_index_db/ ở ĐÚNG path
khai báo trong .env, .env lại cần đứng cùng cấp main.py để các file KHÁC
đọc đúng -> khuyến nghị vẫn chạy từ System_ChatBox/ cho nhất quán):
  python tools/batch_test_category_mapper.py
Cần đã build_category_index.py xong (đọc category_index_db).
"""

import os
import sys

# MỚI — thêm thư mục CHA (System_ChatBox/) vào sys.path, vì file này giờ
# nằm trong tools/ (1 cấp con), còn category_mapper.py ở thư mục cha.
# __file__ = đường dẫn file này -> dirname 2 lần lùi về đúng System_ChatBox/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from category_mapper import resolve_leaf_categories

# Danh sách từ phổ biến khi mua sắm Home & Kitchen — gộp từ nhiều nhóm: dụng cụ
# bếp, đồ dùng phòng khách/ngủ, đồ trang trí, đồ vệ sinh, thiết bị điện gia dụng.
# KHÔNG cần đầy đủ mọi từ có thể có — chỉ cần đủ RỘNG để bắt được phần lớn lỗi
# trong 1 lượt, còn từ nào lạ hơn thì vẫn có thể lọt lưới, phát hiện tiếp qua
# /chat như bình thường.
TEST_WORDS = [
    # Dụng cụ nấu ăn
    "chảo", "ấm đun nước", "nồi cơm điện", "lò nướng", "lò vi sóng",
    "máy xay sinh tố", "máy pha cà phê", "nồi chiên không dầu",
    "dao làm bếp", "kéo cắt", "thớt", "muỗng", "đũa", "vá múc canh",
    "bát", "chén", "đĩa", "ly uống nước", "cốc", "bình giữ nhiệt",
    "hộp đựng thực phẩm", "rổ rá", "khay nướng bánh",
    # Đồ phòng khách / phòng ngủ
    "ghế sofa", "bàn ăn", "kệ tivi", "tủ quần áo", "giường ngủ",
    "gối ôm", "chăn mền", "ga trải giường", "rèm cửa", "thảm trải sàn",
    "đèn bàn", "quạt điện", "gương soi", "kệ sách",
    # Vệ sinh / dọn dẹp
    "máy hút bụi", "chổi quét nhà", "thùng rác", "móc treo quần áo",
    "giỏ đựng đồ giặt", "khăn tắm", "thảm chùi chân",
    # Trang trí
    "bình hoa", "nến thơm", "tranh treo tường", "đồng hồ treo tường",
]


def main():
    print(f"Kiểm tra {len(TEST_WORDS)} từ...\n")
    empty_count = 0
    for word in TEST_WORDS:
        result = resolve_leaf_categories(word)
        marker = "  (rỗng — bình thường, không phải lỗi)" if not result else ""
        print(f"{word:20} -> {result}{marker}")
        if not result:
            empty_count += 1
    print(f"\nTổng: {len(TEST_WORDS)} từ, {empty_count} từ ra rỗng (bình thường), "
          f"{len(TEST_WORDS) - empty_count} từ có kết quả (tự rà xem có hợp lý không).")


if __name__ == "__main__":
    main()