"""
color_mapper.py
Ánh xạ tên MÀU tiếng Việt (do NeedExtractor trích xuất) sang tiếng Anh để
so khớp LIKE với cột `color` trong bảng `products` (dữ liệu gốc Amazon,
tiếng Anh — cùng nguyên nhân lệch ngôn ngữ như category, nhưng miền giá trị
nhỏ và tương đối cố định nên dùng bảng dịch tay thay vì vector index.

GHI CHÚ QUAN TRỌNG — giới hạn đã biết của dữ liệu (không thuộc phạm vi sửa
bằng mô-đun này): kiểm tra DISTINCT color thật cho thấy cột `color` trong
catalog Bộ 1 khá NHIỄU — lẫn nhiều giá trị KHÔNG PHẢI màu (vd "Grain Mill",
"Round", "Spiderman", "Rosewood 3 in 1 Grill Brush", "Original Version"...),
nhiều khả năng do lỗi ánh xạ cột lúc crawl/clean dữ liệu gốc (Data_Cleaning,
Bộ 1 — nhóm khác xử lý). Việc dịch VI->EN ở đây CHỈ giải quyết được phần
lệch ngôn ngữ, KHÔNG giải quyết được phần nhiễu dữ liệu đó. Vì color là
OPTIONAL_SOFT_SLOT (không bắt buộc), nhóm chấp nhận rủi ro recall thấp hơn
mong muốn cho tiêu chí này, và nên ghi rõ giới hạn này khi viết mục hạn chế
ở Chương 5 / phần "vấn đề kỹ thuật" của báo cáo.

Dùng LIKE (không phải "="): để bắt được các giá trị ghép như
"Black/Stainless Steel", "White/Yellow", "Brown and White" — vẫn coi là khớp
nếu chứa đúng tên màu cần tìm.
"""

import re
import unicodedata
from typing import Optional

# Bảng dịch tay — chỉ các màu PHỔ BIẾN thật sự xuất hiện trong dữ liệu (đối
# chiếu với DISTINCT color thật đã kiểm tra), sắp theo độ dài khoá giảm dần
# để ưu tiên khớp cụm dài/cụ thể hơn trước (vd "xám đậm" nên khớp trước "xám"
# nếu sau này bổ sung — hiện tại đã để sẵn cấu trúc cho việc mở rộng).
VI_TO_EN_COLOR = {
    "trắng": "White",
    "đen": "Black",
    "xám": "Gray",
    "xám đậm": "Dark Gray",
    "bạc": "Silver",
    "vàng": "Yellow",
    "đỏ": "Red",
    "hồng": "Pink",
    "xanh dương": "Blue",
    "xanh da trời": "Blue",
    "xanh lá": "Green",
    "nâu": "Brown",
    "be": "Cream",
    "kem": "Cream",
    "trong suốt": "Clear",
    "trong": "Transparent",
    "nhiều màu": "Multi",
    "đa sắc": "Multicolor",
    "tự nhiên": "Natural",
    "gỗ tự nhiên": "Natural Wood",
    "hồng ngọc": "Ruby",
    "đồng": "Brass",
    "ngà": "ivory",
    "hồng phấn": "Rose",
}

# Sắp xếp khoá theo độ dài giảm dần MỘT LẦN khi import — để khi so khớp,
# cụm dài/cụ thể hơn ("xám đậm") được kiểm tra trước cụm ngắn ("xám"),
# tránh khớp nhầm sớm.
_SORTED_KEYS = sorted(VI_TO_EN_COLOR.keys(), key=len, reverse=True)


def _normalize(text: str) -> str:
    """Bỏ dấu tiếng Việt + về chữ thường, để so khớp không phân biệt hoa/thường
    và không phân biệt cách gõ dấu (vd người dùng gõ không dấu)."""
    text = text.strip().lower()
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# Bảng tra cứu đã chuẩn hoá (bỏ dấu) song song với bảng gốc có dấu, để chấp
# nhận cả 2 kiểu người dùng gõ.
_NORMALIZED_LOOKUP = {_normalize(k): v for k, v in VI_TO_EN_COLOR.items()}
_NORMALIZED_SORTED_KEYS = sorted(_NORMALIZED_LOOKUP.keys(), key=len, reverse=True)


def resolve_color(vi_color_text: str) -> Optional[str]:
    """Trả về từ khoá màu TIẾNG ANH tương ứng để dùng trong LIKE '%...%',
    hoặc None nếu không nhận diện được (khi đó rule_based_filter nên BỎ QUA
    điều kiện lọc màu — đúng tinh thần soft slot, không tự bịa/ép giá trị)."""
    if not vi_color_text or not vi_color_text.strip():
        return None

    normalized_input = _normalize(vi_color_text)

    # Khớp trực tiếp nguyên cụm trước (nhanh, chính xác nhất).
    if normalized_input in _NORMALIZED_LOOKUP:
        return _NORMALIZED_LOOKUP[normalized_input]

    # Khớp theo cụm con — phòng trường hợp NeedExtractor trả về câu dài hơn
    # (vd "màu trắng" thay vì "trắng"). BẮT BUỘC dùng ranh giới từ (\b...\b),
    # KHÔNG dùng "in" (substring thô) — đã phát hiện lỗi thật: "khong biet"
    # bị khớp nhầm thành "hồng"/Pink vì chuỗi con "hong" nằm lọt trong "khong".
    for key in _NORMALIZED_SORTED_KEYS:
        if re.search(rf"\b{re.escape(key)}\b", normalized_input):
            return _NORMALIZED_LOOKUP[key]

    print(f"[color_mapper] Không nhận diện được màu '{vi_color_text}' trong bảng dịch — "
          f"bỏ qua lọc màu ở lượt này (không tự đoán).")
    return None


if __name__ == "__main__":
    for demo in ["trắng", "màu đen", "Xám", "khong biet", "xanh dương nhạt"]:
        print(demo, "->", resolve_color(demo))