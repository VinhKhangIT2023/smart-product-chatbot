"""
currency.py
Quy đổi USD <-> VND bằng TỶ GIÁ CỐ ĐỊNH — quyết định CHỦ ĐÍCH của nhóm,
không phải hạn chế kỹ thuật. Lý do (đã thống nhất khi thảo luận):

  1. Dữ liệu giá gốc trong catalog là USD (Amazon), giữ nguyên, không đổi
     trong MySQL — xem KET_QUA_V3.md: "Dữ liệu giá vẫn là USD."
  2. Người dùng Việt Nam nói giá theo thói quen VND (vd "dưới 500k").
  3. Tỷ giá thời gian thực (qua API bên ngoài) có 2 vấn đề cho hệ thống
     quy mô khóa luận:
       (a) Thêm 1 lệnh gọi mạng ngoài mỗi khi cần tỷ giá -> có thể lỗi/
           timeout, giảm độ ổn định hệ thống (đi ngược NFR-04 "Độ bền khi
           lỗi" đã đặt ra).
       (b) NGAY CẢ KHI lấy tỷ giá tại thời điểm build/khởi động, tỷ giá đó
           vẫn "đông cứng" (frozen) tại đúng thời điểm lấy — KHÔNG thật sự
           phản ánh "thời gian thực" trong suốt phiên người dùng đang dùng
           thử, trừ khi gọi API lại ở MỖI lượt chat (tốn thêm độ trễ, thêm
           1 điểm có thể lỗi, không đáng cho quy mô demo/khóa luận này).
  -> Dùng tỷ giá CỐ ĐỊNH, đặt ở 1 NƠI DUY NHẤT (file này) — nếu cần cập
     nhật tỷ giá sau này (hoặc thật sự muốn tích hợp API), chỉ cần sửa
     đúng 1 chỗ, mọi nơi dùng (main.py, need_extractor.py, ...) tự động
     nhất quán theo.

Mọi giá trị VND quy đổi ra đều chỉ mang tính THAM KHẢO (ước tính theo tỷ
giá cố định), KHÔNG phải giá niêm yết chính thức — khi hiển thị cho người
dùng, nên nói rõ đây là "tỷ giá tham khảo", tránh khẳng định quá mức (đúng
tinh thần "không khẳng định điều chưa kiểm chứng" xuyên suốt khóa luận).
"""

# CHỐT: 1 USD = 26.000 VND (theo quyết định của nhóm). Sửa DUY NHẤT tại
# đây nếu cần cập nhật tỷ giá — không sửa rải rác ở file khác.
USD_TO_VND_RATE = 26_000


def usd_to_vnd(usd: float) -> float:
    """Quy đổi 1 số tiền USD sang VND theo tỷ giá cố định ở trên."""
    return usd * USD_TO_VND_RATE


def vnd_to_usd(vnd: float) -> float:
    """Quy đổi 1 số tiền VND sang USD theo tỷ giá cố định ở trên."""
    return vnd / USD_TO_VND_RATE


def round_vnd(vnd: float, nearest: int = 1000) -> float:
    """Làm tròn số tiền VND tới bội số gần nhất (mặc định tới nghìn đồng,
    quy ước thường gặp khi hiển thị giá tiếng Việt — vd 259.740đ ->
    260.000đ), chỉ dùng cho HIỂN THỊ, không dùng số đã làm tròn này để lọc
    hay tính toán lại (tránh sai lệch tích lũy do làm tròn)."""
    if nearest <= 0:
        return vnd
    return round(vnd / nearest) * nearest


if __name__ == "__main__":
    print(f"Tỷ giá đang dùng: 1 USD = {USD_TO_VND_RATE:,} VND")
    for usd in [1, 10, 25.5, 100]:
        vnd = usd_to_vnd(usd)
        print(f"  {usd} USD -> {vnd:,.0f} VND (làm tròn nghìn: {round_vnd(vnd):,.0f} VND)")
    for vnd in [500_000, 1_000_000, 26_000]:
        usd = vnd_to_usd(vnd)
        print(f"  {vnd:,} VND -> {usd:.2f} USD")