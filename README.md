# Hệ thống Chatbot Tư vấn Sản phẩm (CNTT-KLCN202)

Backend chatbot tư vấn sản phẩm hội thoại, kiến trúc RAG kết hợp Rule-based (MySQL) + AI Matching (Chroma vector search) + LLM (Qwen).

## Cấu trúc thư mục

```
System_ChatBox/
├── .env.example              # Mẫu khai báo biến môi trường — copy thành .env rồi điền giá trị thật
├── requirements.txt
├── few_shot_examples.py      # Mẫu phong cách hội thoại cho Qwen
├── session_manager.py        # DialogueService — quyết định hành động mỗi lượt chat
├── need_extractor.py         # NeedExtractor — trích slot nhu cầu từ tin nhắn (gọi Qwen)
├── category_mapper.py        # (MỚI) Dịch category tiếng Việt -> leaf_category tiếng Anh
│                             #   thật trong catalog, bằng vector similarity (không gọi Qwen)
├── color_mapper.py           # (MỚI) Dịch tên màu tiếng Anh -> tiếng Việt, bảng tĩnh
├── import_to_mysql.py        # Nạp catalog sản phẩm (CSV) vào MySQL
├── build_vector_index.py     # Mã hóa embedding_text sản phẩm -> Chroma (chạy 1 lần, offline)
├── build_category_index.py   # (MỚI) Mã hóa tên leaf_category -> Chroma riêng (chạy 1 lần,
│                             #   offline) — nguồn dữ liệu cho category_mapper.py
├── vector_search.py          # AI Matching — xếp hạng ngữ nghĩa trên Chroma (tầng 2)
├── test_qwen_api.py          # Test kết nối Qwen API
└── main.py                   # FastAPI backend chính (endpoint /chat) — dùng
                              #   category_mapper/color_mapper trong rule_based_filter()
```

## Lưu ý quan trọng: KHÔNG có dữ liệu (Dataset/) trong repo này

Thư mục `Dataset/` (catalog sản phẩm CSV, dữ liệu hội thoại JSON) đã bị loại khỏi repo (`.gitignore`) vì:
- Một số file vượt quá giới hạn 100MB/file của GitHub.
- Đây là dữ liệu, không phải mã nguồn, và có thể tái tạo lại bằng script.
- Một phần dữ liệu (catalog Home & Kitchen) do thành viên khác trong nhóm xử lý.

Để chạy được hệ thống, bạn cần tự có file catalog CSV (`amazon_home_kitchen_catalog_ready.csv`) và làm theo bước Setup dưới đây.

## ⚠️ Vấn đề đã phát hiện và đã sửa: lệch ngôn ngữ category/color (Việt/Anh)

**Hiện tượng (bản cũ):** `need_extractor.py` trích xuất `category`/`color` bằng
**tiếng Việt** (theo đúng few-shot đã dạy, vd `"thảm trải sàn"`, `"đỏ"`), nhưng
catalog gốc (`amazon_home_kitchen_catalog_ready.csv`) là dữ liệu Amazon
**tiếng Anh** (`leaf_category` = `"Area Rugs"`, `color` = `"Red"`...) — câu
lọc SQL `LIKE '%thảm trải sàn%'` trong `rule_based_filter()` không bao giờ
khớp, khiến `SEARCH_PRODUCTS` luôn trả về **0 sản phẩm**, bất kể nhu cầu
người dùng là gì.

**Đã xác nhận thêm:** cột `main_category` trong catalog **chỉ có đúng 1 giá
trị duy nhất** (`"Home_and_Kitchen"`) cho toàn bộ 15.714 dòng — lọc theo cột
này không có tác dụng thu hẹp, chỉ `leaf_category` mới thật sự phân loại
được sản phẩm.

**Cách đã sửa:**
- **category** (bắt buộc/hard slot): thêm `category_mapper.py` + script
  offline `build_category_index.py` — xây một chỉ mục vector **nhỏ, riêng**
  chỉ chứa tên các `leaf_category` thật lấy trực tiếp từ MySQL, dùng chung
  model embedding đa ngôn ngữ đã có (`paraphrase-multilingual-MiniLM-L12-v2`).
  Khi chat, câu tiếng Việt được encode và tìm các `leaf_category` tiếng Anh
  gần nghĩa nhất (có thể trả về NHIỀU category, vd "nồi" → `Stockpots`,
  `Saucepans`, `Dutch Ovens`...), dùng làm điều kiện `leaf_category IN (...)`.
  **Không tốn thêm lần gọi Qwen nào** — vẫn giữ đúng 2 lần gọi Qwen/lượt chat
  đã thiết kế từ đầu.
- **color** (tuỳ chọn/soft slot): thêm `color_mapper.py` — bảng dịch tay
  VI→EN tĩnh cho các màu phổ biến (miền giá trị nhỏ, không cần vector index
  như category).
- Nếu không map được category/color nào đủ tin cậy, hệ thống **bỏ qua điều
  kiện lọc đó** thay vì tự đoán hoặc tự nới các ràng buộc khác — có log cảnh
  báo rõ ràng ở console để dễ debug.

**Giới hạn còn lại, CHƯA giải quyết được bằng code (thuộc chất lượng dữ liệu
nguồn, không phải lỗi logic):** kiểm tra `DISTINCT color` thật cho thấy cột
`color` trong catalog Bộ 1 khá nhiễu — lẫn nhiều giá trị không phải màu (vd
`"Grain Mill"`, `"Round"`, `"Spiderman"`, `"Rosewood 3 in 1 Grill Brush"`),
nhiều khả năng do lỗi ánh xạ cột lúc crawl/clean dữ liệu gốc. `color_mapper.py`
chỉ giải quyết được phần lệch ngôn ngữ, không giải quyết được phần nhiễu này
— cần ghi vào mục hạn chế của báo cáo, và cân nhắc làm sạch lại cột `color`
ở phía Data_Cleaning (Bộ 1) nếu còn thời gian.

## Setup

1. Cài thư viện:
   ```
   pip install -r requirements.txt
   ```

2. Cài MySQL, tạo database:
   ```sql
   CREATE DATABASE klcn_chatbot CHARACTER SET utf8mb4;
   ```

3. Copy `.env.example` thành `.env`, điền giá trị thật (mật khẩu MySQL, API key Qwen — xem hướng dẫn lấy key ở `test_qwen_api.py`).

4. Nạp catalog sản phẩm vào MySQL (sửa đường dẫn CSV trong file nếu cần):
   ```
   python import_to_mysql.py
   ```

5. Build vector index cho sản phẩm (chạy 1 lần, hoặc khi catalog đổi):
   ```
   python build_vector_index.py
   ```

6. **(MỚI)** Build vector index cho tên category (chạy 1 lần, hoặc khi có
   category mới trong catalog — cần chạy SAU bước 4 vì đọc trực tiếp từ MySQL):
   ```
   python build_category_index.py
   ```
   Kiểm tra nhanh chất lượng ánh xạ trước khi chạy cả hệ thống:
   ```
   python category_mapper.py
   ```

7. Test kết nối Qwen:
   ```
   python test_qwen_api.py
   ```

8. Chạy hệ thống:
   ```
   uvicorn main:app --reload
   ```
   Mở `http://127.0.0.1:8000/docs` để test qua Swagger UI.

## Kiến trúc

Xem chi tiết trong báo cáo `BaoCao_HeThongChatbot_SystemChatBox.docx` (kiến trúc RAG 2 giai đoạn: Indexing offline + Retrieval/Generation runtime, sơ đồ minh họa, mô tả từng file, kết quả thực nghiệm — **CẦN CẬP NHẬT** phần này để phản ánh `category_mapper.py`/`color_mapper.py`/`build_category_index.py` và thêm mục "vấn đề category/color tiếng Anh/Việt" vào khó khăn kỹ thuật).

## Đề tài
Hệ thống tư vấn sản phẩm thông minh theo nhu cầu người dùng — CNTT-KLCN202.