"""
color_mapper.py
Ánh xạ tên MÀU tiếng Việt (do NeedExtractor trích xuất) sang tiếng Anh để
so khớp với cột `color` trong bảng `products`.

--- ĐÃ VIẾT LẠI HOÀN TOÀN (dataset v3) ---
Bản trước dùng danh sách màu chuẩn TỰ ĐẶT (đoán từ khảo sát DISTINCT color
thô). Từ dataset v3 (`Dataset1 Home & Kitchen 40k v3/Data_Cleaning/`), nhóm
xử lý dữ liệu (Bộ 1) đã tự xây 1 pipeline làm sạch màu RIÊNG, bài bản hơn
hẳn: mỗi sản phẩm có `color` (giá trị CHUẨN cuối cùng), `color_raw` (giá trị
gốc), `colors_json` (mảng JSON các màu chuẩn), `color_status` (6 loại:
recognized_single/multi/with_metadata, missing, non_color, needs_review),
`color_rule`, `color_needs_review` (cờ 0/1).

Cột `color` trong dataset v3 CHỈ CHỨA ĐÚNG 23 GIÁ TRỊ CHUẨN SAU (đã kiểm tra
thật bằng cách tách toàn bộ giá trị `color`, kể cả các dòng đa màu ghép bằng
" | "):
  Beige, Black, Blue, Bronze, Brown, Clear, Copper, Cream, Gold, Gray,
  Green, Ivory, Multicolor, Orange, Pink, Purple, Red, Rose Gold, Silver,
  Teal, Turquoise, White, Yellow

QUAN TRỌNG — khác biệt so với bản trước: pipeline của Bộ 1 CHỦ ĐÍCH coi
"Stainless Steel", "Chrome", "Bamboo" là "non_color" (KHÔNG PHẢI màu, mà là
chất liệu/hoàn thiện bề mặt) — ngược lại bản color_mapper.py TRƯỚC ĐÂY của
module này từng tự thêm "Stainless Steel"/"Chrome"/"Brass" vào danh sách màu
(SAI, đã sửa). Vì "Brass" không tồn tại trong 23 giá trị chuẩn thật, mọi
input tiếng Việt liên quan kim loại giờ trỏ về "Copper" (đúng nghĩa "đồng")
hoặc "Bronze" ("đồng thau") — 2 giá trị THẬT SỰ có trong dữ liệu, thay vì
"Brass" (không tồn tại, dịch xong cũng không bao giờ khớp được sản phẩm nào).

ĐA MÀU: nhiều sản phẩm có `color` dạng ghép " | " (vd "Black | Gold") — hàm
resolve_color() ở đây CHỈ trả về 1 tên chuẩn duy nhất (đúng như trước), việc
so khớp AN TOÀN với cả giá trị đơn lẫn giá trị ghép được xử lý ở main.py
(rule_based_filter dùng CONCAT(' | ', color, ' | ') LIKE '% | X | %' — KHÔNG
dùng LIKE '%X%' đơn giản, để tránh khớp nhầm substring, vd tìm "Gold" đơn
thuần không được lọt vào "Rose Gold").

GHI CHÚ VỀ needs_review: cột `color` trong dataset v3 đã TỰ NULL HÓA cho mọi
dòng ở trạng thái needs_review/missing/non_color — tức khi lọc SQL theo
`color`, các dòng này TỰ ĐỘNG bị loại (NULL không bao giờ khớp LIKE), KHÔNG
cần xử lý gì thêm ở tầng ứng dụng cho việc này.
"""

import re
import unicodedata
from typing import Optional

# Bảng dịch tay VI -> EN, CHỈ trỏ tới các giá trị THẬT SỰ TỒN TẠI trong cột
# `color` của dataset v3 (xem danh sách 23 giá trị ở docstring trên) — dịch
# sang giá trị không tồn tại thì lọc SQL sẽ luôn ra 0 kết quả một cách vô ích.
VI_TO_EN_COLOR = {
    "trắng": "White",
    "đen": "Black",
    "xám": "Gray",
    "bạc": "Silver",
    "vàng": "Yellow",
    "vàng kim": "Gold",
    "vàng hồng": "Rose Gold",
    "đỏ": "Red",
    "hồng": "Pink",
    "xanh dương": "Blue",
    "xanh da trời": "Blue",
    "xanh lá": "Green",
    "xanh lục lam": "Teal",
    "xanh mòng két": "Teal",
    "xanh ngọc": "Turquoise",
    "ngọc lam": "Turquoise",
    "nâu": "Brown",
    "be": "Beige",
    "kem": "Cream",
    "trong suốt": "Clear",
    "trong": "Clear",
    "nhiều màu": "Multicolor",
    "đa sắc": "Multicolor",
    "ngà": "Ivory",
    "ngà voi": "Ivory",
    "đồng": "Copper",       # tiếng Việt "đồng" = kim loại đồng -> Copper (SỬA so với bản
                             # trước, từng trỏ nhầm sang "Brass" — Brass không tồn tại
                             # trong dữ liệu thật của dataset v3).
    "đồng thau": "Bronze",  # phân biệt với "đồng" (Copper) — đồng thau/hợp kim -> Bronze.
    "tím": "Purple",
    "cam": "Orange",
}

# Sắp theo độ dài giảm dần — ưu tiên khớp cụm dài/cụ thể hơn trước (vd
# "đồng thau" phải được kiểm tra TRƯỚC "đồng", nếu không "đồng" sẽ khớp
# nhầm trước do là cụm con của "đồng thau").
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
    """Trả về 1 trong 23 giá trị màu CHUẨN thật sự tồn tại trong cột `color`
    của dataset v3, hoặc None nếu không nhận diện được (khi đó rule_based_filter
    nên BỎ QUA điều kiện lọc màu — đúng tinh thần soft slot, không tự bịa/ép
    giá trị)."""
    if not vi_color_text or not vi_color_text.strip():
        return None

    normalized_input = _normalize(vi_color_text)

    # Khớp trực tiếp nguyên cụm trước (nhanh, chính xác nhất).
    if normalized_input in _NORMALIZED_LOOKUP:
        return _NORMALIZED_LOOKUP[normalized_input]

    # Khớp theo cụm con — phòng trường hợp NeedExtractor trả về câu dài hơn
    # (vd "màu trắng" thay vì "trắng"). BẮT BUỘC dùng ranh giới từ (\b...\b),
    # KHÔNG dùng "in" (substring thô) — bản trước đã có lỗi thật: "khong biet"
    # bị khớp nhầm thành "hồng"/Pink vì chuỗi con "hong" nằm lọt trong "khong".
    for key in _NORMALIZED_SORTED_KEYS:
        if re.search(rf"\b{re.escape(key)}\b", normalized_input):
            return _NORMALIZED_LOOKUP[key]

    print(f"[color_mapper] Không nhận diện được màu '{vi_color_text}' trong bảng dịch — "
          f"bỏ qua lọc màu ở lượt này (không tự đoán).")
    return None


if __name__ == "__main__":
    for demo in ["trắng", "màu đen", "Xám", "khong biet", "xanh dương nhạt",
                 "đồng", "đồng thau", "vàng hồng", "xanh ngọc"]:
        print(demo, "->", resolve_color(demo))