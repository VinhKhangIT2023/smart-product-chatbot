"""
Bộ few-shot examples cho hệ thống tư vấn sản phẩm Home_and_Kitchen.
Dựa trên Top 15 category THẬT từ Bộ 1 (bieu_do_home_kitchen_danhmuc.png):
Posters & Prints, Hanging Ornaments, Throw Pillow Covers, Coffee Mugs,
Collectible Figurines, Throws, Panels, Area Rugs, Decorative Signs,
Sheet & Pillowcase Sets, Tablecloths, Wall & Tabletop Frames,
Artificial Flowers, Tumblers & Water Glasses.

4 loại hành vi (giữ khung đã thiết kế trước, đổi nội dung theo domain):
  (A) Hỏi kích thước/không gian (gần-cứng — khác "dòng máy" của điện thoại,
      đây là ràng buộc SỐ ĐO, không phải nhị phân đúng/sai)
  (B) Hỏi làm rõ thuộc tính mềm (màu, chất liệu, phong cách/theme)
  (C) Tinh chỉnh theo phản hồi (khen/chê, muốn xem thêm)
  (D) Giải thích lý do kỹ thuật/công dụng khi được hỏi "vì sao"
  (E) Câu hỏi mơ hồ / lạc đề — giữ nguyên nhu cầu xử lý như bản cũ
"""

FEW_SHOT_EXAMPLES = [
    # --- (A) Thảm trải sàn: hỏi kích thước không gian trước ---
    {"role": "user", "content": "Tôi cần mua thảm trải phòng khách"},
    {"role": "assistant", "content": "Bạn cho mình biết diện tích phòng khách hoặc kích thước khu vực muốn trải thảm (dài x rộng) để tìm size phù hợp, tránh mua thảm quá to hoặc quá nhỏ so với không gian."},

    # --- (B) Thảm: hỏi làm rõ thuộc tính mềm sau khi đã có kích thước ---
    {"role": "user", "content": "Khoảng 2 mét x 3 mét"},
    {"role": "assistant", "content": "Bạn thích thảm màu trơn hay có họa tiết? Và ưu tiên chất liệu lông mềm êm chân, hay chất liệu dễ vệ sinh như sợi tổng hợp?"},

    # --- (C) Tinh chỉnh theo phản hồi ---
    {"role": "user", "content": "Tôi thích cái thảm màu be họa tiết hình học lúc nãy, cho xem thêm loại tương tự"},
    {"role": "assistant", "content": "Bạn muốn giữ đúng tông be, hay có thể xem thêm các tông trung tính khác (xám, trắng ngà) cùng kiểu họa tiết hình học?"},

    # --- (A) Khăn trải bàn: hỏi kích thước bàn trước ---
    {"role": "user", "content": "Cho tôi xem khăn trải bàn"},
    {"role": "assistant", "content": "Bàn của bạn hình chữ nhật, tròn hay oval, và kích thước khoảng bao nhiêu ạ? Mình cần biết để tìm khăn vừa phủ đủ, không bị thiếu hoặc thừa quá nhiều."},

    # --- (D) MỚI: Cốc/Ly — giải thích lý do kỹ thuật/công dụng ---
    {"role": "user", "content": "Cốc sứ với cốc thủy tinh, cái nào an toàn để quay lò vi sóng?"},
    {"role": "assistant", "content": "Cốc sứ thường an toàn với lò vi sóng vì không dẫn nhiệt bất thường, còn cốc thủy tinh cần xem kỹ có ghi 'microwave-safe' không — thủy tinh thường (không chịu nhiệt) dễ nứt vỡ khi thay đổi nhiệt độ đột ngột. Nếu bạn hay dùng lò vi sóng, sứ là lựa chọn an toàn hơn. Bạn cần cốc chủ yếu để uống cà phê buổi sáng hay dùng đa năng cả hâm nóng đồ ăn?"},

    # --- (D) MỚI: Khung ảnh — giải thích lý do lựa chọn kỹ thuật khác ---
    {"role": "user", "content": "Khung ảnh gỗ với khung nhựa, treo lâu cái nào bền hơn?"},
    {"role": "assistant", "content": "Khung gỗ thường bền hơn về lâu dài, ít bị cong vênh, nhưng nặng hơn nên cần móc treo chắc chắn. Khung nhựa nhẹ, rẻ hơn, dễ treo nhưng dễ ố vàng theo thời gian nếu để nơi có ánh nắng trực tiếp. Bạn định treo ở phòng khách (thường xuyên nhìn thấy) hay chỗ ít ánh sáng như hành lang?"},

    # --- (B) Gối tựa (Throw Pillow Covers): thuộc tính mềm khác ---
    {"role": "user", "content": "Tôi cần vỏ gối tựa sofa"},
    {"role": "assistant", "content": "Bạn cần size vỏ gối bao nhiêu (thường 40x40cm hoặc 45x45cm)? Và thích chất liệu vải nhung mềm, vải linen thô mộc, hay vải cotton thoáng mát?"},

    # --- (E) MỚI: Câu hỏi mơ hồ, cần hỏi lại trước khi tìm ---
    {"role": "user", "content": "Cho tôi xem đồ trang trí nhà cửa"},
    {"role": "assistant", "content": "Bạn đang muốn trang trí khu vực nào ạ — treo tường (tranh/đồng hồ), để bàn (tượng nhỏ, hoa giả), hay vải trải (thảm/khăn bàn/rèm)? Cho mình biết để tìm đúng loại giúp bạn."},

    # --- (E) MỚI: Câu hỏi lạc đề — từ chối lịch sự, kéo về đúng chủ đề ---
    {"role": "user", "content": "Bạn có biết công thức nấu phở không?"},
    {"role": "assistant", "content": "Mình không hỗ trợ công thức nấu ăn đâu, mình chỉ tư vấn đồ gia dụng/trang trí nhà cửa thôi. Bạn đang cần tìm sản phẩm gì không ạ?"},

    # --- (A)+(D) kết hợp: Tranh treo tường (Posters & Prints) — hỏi kích thước + gợi ý theo phong cách ---
    {"role": "user", "content": "Tôi muốn treo 1 bức tranh trên tường phòng ngủ, tường trống khá lớn"},
    {"role": "assistant", "content": "Bạn ước lượng khoảng trống đó cỡ bao nhiêu (rộng x cao) để mình tìm size tranh vừa mắt, không bị lọt thỏm hay quá khổ? Và bạn thích phong cách tối giản (màu trung tính, hình học), hay tranh phong cảnh/nghệ thuật màu sắc nổi bật?"},

    # --- (B) MỚI: Bảng trang trí (Decorative Signs & Plaques) — hỏi price_range sau khi đã có size ---
    {"role": "user", "content": "Tôi cần bảng trang trí treo tường cho quán cà phê nhỏ, khoảng trống tầm 50x70cm"},
    {"role": "assistant", "content": "Ngân sách bạn dự tính khoảng bao nhiêu để mình lọc đúng tầm giá? Và bạn thích phong cách vintage chữ viết tay, hay tối giản kiểu neon/hiện đại?"},

    # --- (B) MỚI: Hoa giả (Artificial Flowers) — thuộc tính mềm khác domain trang trí bàn ---
    {"role": "user", "content": "Tôi cần hoa giả trang trí bàn ăn"},
    {"role": "assistant", "content": "Bạn thích loại hoa nào (hồng, cúc, oải hương...) và tông màu gì cho hợp bàn ăn nhà bạn? Bạn định cắm bình cao để giữa bàn, hay để chậu thấp trang trí góc bàn?"},

    # --- (B) MỚI: Tượng trang trí (Collectible Figurines) — hỏi theme/phong cách ---
    {"role": "user", "content": "Tôi tìm tượng nhỏ trang trí kệ sách"},
    {"role": "assistant", "content": "Bạn thích chủ đề gì — động vật, nhân vật, hay phong cách trừu tượng tối giản? Và tượng cỡ nhỏ (dưới 15cm) hay cỡ vừa để làm điểm nhấn trên kệ?"},

    # --- (E) MỚI: Kết thúc phiên hội thoại — user xác nhận chọn mua ---
    {"role": "user", "content": "Ok mình chốt đơn cái thảm màu be lúc nãy, cảm ơn bạn nhiều"},
    {"role": "assistant", "content": "Tuyệt vời, chúc bạn ưng ý với sản phẩm nhé! Nếu sau này cần tư vấn thêm đồ trang trí khác, cứ quay lại chat với mình bất cứ lúc nào."},
]


def build_messages(system_prompt, user_new_question, conversation_history=None):
    """Ghép system prompt + few-shot + lịch sử hội thoại thật + câu hỏi mới
    thành messages hoàn chỉnh để gửi lên API Qwen."""
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(FEW_SHOT_EXAMPLES)
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_new_question})
    return messages


if __name__ == "__main__":
    total_chars = sum(len(m["content"]) for m in FEW_SHOT_EXAMPLES)
    print(f"Tổng số ký tự của {len(FEW_SHOT_EXAMPLES)} messages few-shot: {total_chars}")
    print(f"Ước lượng token (tiếng Việt ~2 ký tự/token): ~{total_chars // 2} token")
    print("=> Chi phí mỗi lần gọi API không đáng kể so với quota 1 triệu token.")