"""
session_manager.py
Quản lý phiên hội thoại nhiều lượt cho hệ thống tư vấn sản phẩm Home_and_Kitchen.

Khung logic: SessionState (trạng thái từng phiên) + decide_next_action()
(quyết định hành động tiếp theo mỗi lượt) + filter_out_already_shown()
(tránh lặp gợi ý — vấn đề chính khiến % positive không tăng sau turn 3
trong phân tích Bộ 2: ConvRec Shopping).

REQUIRED_HARD_SLOTS = ["category", "size_space"]
  - category: bắt buộc, không có thì không biết tìm loại sản phẩm gì.
  - size_space: gần-cứng, nhưng KHÔNG ép nhập số đo chính xác — chỉ cần
    đã hỏi 1 lần là coi như "resolved" (kể cả khi người dùng trả lời
    "không quan tâm"/mô tả mơ hồ). Việc đánh giá độ phù hợp kích thước
    thực tế được để mở cho bước AI Matching / LLM xử lý từ mô tả tự
    nhiên, KHÔNG so sánh số đo tuyệt đối cứng nhắc ở tầng session.

PRIORITY_SOFT_SLOTS = ["price_range"] — hỏi trước các soft slot khác
  vì ảnh hưởng nhiều đến kết quả, nhưng KHÔNG bắt buộc (thiếu vẫn
  search được, chỉ là không lọc theo ngân sách).

OPTIONAL_SOFT_SLOTS = ["color", "material", "style", "brand"] — hỏi
  dần dần theo turn, không hỏi dồn hết 1 lúc.

File này KHÔNG cần đăng nhập/API key gì cả — chạy độc lập được ngay.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


# ---------- Cấu hình slot theo domain Home_and_Kitchen ----------
REQUIRED_HARD_SLOTS = ["category", "size_space"]
PRIORITY_SOFT_SLOTS = ["price_range"]
OPTIONAL_SOFT_SLOTS = ["color", "material", "style", "brand"]

ALL_SLOTS = REQUIRED_HARD_SLOTS + PRIORITY_SOFT_SLOTS + OPTIONAL_SOFT_SLOTS

# Số soft slot tối đa hỏi thêm trong 1 phiên trước khi ép chuyển sang
# SEARCH_PRODUCTS, để tránh hỏi dồn quá nhiều lượt (theo insight Bộ 2:
# lợi ích không tăng ổn định sau ~3 lượt).
MAX_SOFT_SLOT_QUESTIONS = 2

# Từ khóa tín hiệu kết thúc phiên (đơn giản, dựa từ khóa — có thể nâng
# cấp bằng LLM classify sau).
END_SESSION_KEYWORDS = [
    "cảm ơn", "thế thôi", "vậy thôi", "để tôi suy nghĩ", "thôi khỏi",
    "mua cái này", "đặt hàng", "chốt đơn", "không cần nữa",
]


class NextAction(str, Enum):
    ASK_HARD_SLOT = "ASK_HARD_SLOT"
    ASK_SOFT_SLOT = "ASK_SOFT_SLOT"
    SEARCH_PRODUCTS = "SEARCH_PRODUCTS"
    END_SESSION = "END_SESSION"


@dataclass
class SessionState:
    session_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    history: List[Dict[str, str]] = field(default_factory=list)  # [{role, content}]
    turn_count: int = 0

    # Slot nhu cầu
    slots: Dict[str, Any] = field(default_factory=lambda: {k: None for k in ALL_SLOTS})
    size_space_asked: bool = False        # đã hỏi kích thước/không gian chưa
    soft_slots_asked: int = 0             # đếm số soft slot đã hỏi (để chặn hỏi quá nhiều)
    asked_slots: Set[str] = field(default_factory=set)  # slot nào đã từng hỏi rồi thì không hỏi lại

    # Chống lặp gợi ý sản phẩm giữa các turn
    shown_product_ids: Set[str] = field(default_factory=set)

    ended: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["shown_product_ids"] = list(self.shown_product_ids)
        d["asked_slots"] = list(self.asked_slots)
        return d


class SessionManager:
    """Quản lý toàn bộ phiên đang hoạt động (in-memory).
    Có thể thay bằng lưu xuống MySQL bảng `conversation_sessions`
    (xem import_to_mysql.py) khi tích hợp thật."""

    def __init__(self):
        self._sessions: Dict[str, SessionState] = {}

    # ---------- Quản lý phiên ----------
    def create_session(self) -> SessionState:
        sid = str(uuid.uuid4())
        session = SessionState(session_id=sid)
        self._sessions[sid] = session
        return session

    def get_session(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: Optional[str]) -> SessionState:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        return self.create_session()

    # ---------- Lịch sử hội thoại ----------
    def add_message(self, session_id: str, role: str, content: str) -> SessionState:
        session = self.get_or_create(session_id)
        session.history.append({"role": role, "content": content})
        if role == "user":
            session.turn_count += 1
        session.updated_at = time.time()
        return session

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        session = self.get_session(session_id)
        return session.history if session else []

    # ---------- Cập nhật slot nhu cầu ----------
    def update_slots(self, session_id: str, new_values: Dict[str, Any]) -> SessionState:
        """Ghi đè/bổ sung slot mới trích xuất được từ tin nhắn user.
        Giá trị None bị bỏ qua (không xoá nhu cầu đã thu thập trước đó)."""
        session = self.get_or_create(session_id)
        for key, value in new_values.items():
            if key not in ALL_SLOTS:
                continue
            if value is None:
                continue
            session.slots[key] = value
        session.updated_at = time.time()
        return session

    def mark_size_space_asked(self, session_id: str) -> None:
        """Gọi ngay sau khi hệ thống đã hỏi câu về kích thước/không gian,
        BẤT KỂ người dùng trả lời gì (kể cả 'không quan tâm') — vì slot
        này chỉ cần đã-hỏi để coi là resolved, không ép có số đo."""
        session = self.get_or_create(session_id)
        session.size_space_asked = True

    # ---------- Chống lặp gợi ý ----------
    def filter_out_already_shown(
        self, session_id: str, candidates: List[Dict[str, Any]], id_field: str = "product_id"
    ) -> List[Dict[str, Any]]:
        """Loại các sản phẩm đã gợi ý ở turn trước ra khỏi danh sách ứng viên mới.
        id_field mặc định 'product_id' — khóa chính thật của catalog_ready.csv (Bộ 1)."""
        session = self.get_session(session_id)
        if not session or not session.shown_product_ids:
            return candidates
        return [c for c in candidates if c.get(id_field) not in session.shown_product_ids]

    def mark_as_shown(self, session_id: str, products: List[Dict[str, Any]], id_field: str = "product_id") -> None:
        session = self.get_or_create(session_id)
        for p in products:
            pid = p.get(id_field)
            if pid:
                session.shown_product_ids.add(pid)

    # ---------- Quyết định hành động tiếp theo ----------
    def decide_next_action(self, session_id: str, user_message: str = "") -> Dict[str, Any]:
        """Trả về {"action": NextAction, "slot": Optional[str]}.
        - ASK_HARD_SLOT: cần hỏi category hoặc size_space trước.
        - ASK_SOFT_SLOT: đủ hard slot rồi, hỏi thêm 1 soft slot (ưu tiên
          price_range trước, rồi tới color/material/style/brand), nếu
          chưa vượt MAX_SOFT_SLOT_QUESTIONS.
        - SEARCH_PRODUCTS: đủ điều kiện để tìm & xếp hạng sản phẩm.
        - END_SESSION: phát hiện tín hiệu kết thúc hội thoại.
        """
        session = self.get_or_create(session_id)

        # 1. Kiểm tra tín hiệu kết thúc trước tiên
        lowered = user_message.lower()
        if any(kw in lowered for kw in END_SESSION_KEYWORDS):
            session.ended = True
            return {"action": NextAction.END_SESSION, "slot": None}

        # 2. Hard slot: category trước
        if not session.slots.get("category"):
            session.asked_slots.add("category")
            return {"action": NextAction.ASK_HARD_SLOT, "slot": "category"}

        # 3. Hard slot: size_space (mở — chỉ cần đã hỏi, không ép giá trị)
        if not session.size_space_asked:
            session.asked_slots.add("size_space")
            return {"action": NextAction.ASK_HARD_SLOT, "slot": "size_space"}

        # 4. Soft slot ưu tiên: price_range
        if session.soft_slots_asked < MAX_SOFT_SLOT_QUESTIONS:
            if not session.slots.get("price_range") and "price_range" not in session.asked_slots:
                session.asked_slots.add("price_range")
                session.soft_slots_asked += 1
                return {"action": NextAction.ASK_SOFT_SLOT, "slot": "price_range"}

            # 5. Soft slot còn lại, theo thứ tự cố định, mỗi slot chỉ hỏi 1 lần
            for slot in OPTIONAL_SOFT_SLOTS:
                if not session.slots.get(slot) and slot not in session.asked_slots:
                    session.asked_slots.add(slot)
                    session.soft_slots_asked += 1
                    return {"action": NextAction.ASK_SOFT_SLOT, "slot": slot}

        # 6. Đủ điều kiện để tìm sản phẩm
        return {"action": NextAction.SEARCH_PRODUCTS, "slot": None}

    def get_slots(self, session_id: str) -> Dict[str, Any]:
        session = self.get_session(session_id)
        return session.slots if session else {k: None for k in ALL_SLOTS}


if __name__ == "__main__":
    # Demo nhanh, không cần DB hay API key
    sm = SessionManager()
    s = sm.create_session()
    print("Session:", s.session_id)

    # Turn 1
    sm.add_message(s.session_id, "user", "Tôi cần mua thảm trải phòng khách")
    sm.update_slots(s.session_id, {"category": "thảm trải sàn"})
    action = sm.decide_next_action(s.session_id, "Tôi cần mua thảm trải phòng khách")
    print("Turn 1 ->", action)  # kỳ vọng: ASK_HARD_SLOT / size_space

    # Turn 2 — trả lời kích thước (mở, không ép số)
    sm.add_message(s.session_id, "user", "Khoảng 2m x 3m")
    sm.mark_size_space_asked(s.session_id)
    sm.update_slots(s.session_id, {"size_space": "2m x 3m"})
    action = sm.decide_next_action(s.session_id, "Khoảng 2m x 3m")
    print("Turn 2 ->", action)  # kỳ vọng: ASK_SOFT_SLOT / price_range

    # Turn 3 — bỏ qua giá, hỏi màu
    sm.add_message(s.session_id, "user", "Giá bao nhiêu cũng được")
    action = sm.decide_next_action(s.session_id, "Giá bao nhiêu cũng được")
    print("Turn 3 ->", action)  # kỳ vọng: ASK_SOFT_SLOT / color (do soft_slots_asked=1 < 2)

    print("Slots hiện tại:", sm.get_slots(s.session_id))