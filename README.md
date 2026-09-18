# Hệ thống Chatbot Tư vấn Sản phẩm (CNTT-KLCN202)

Backend chatbot tư vấn sản phẩm hội thoại, kiến trúc RAG kết hợp Rule-based (MySQL) + AI Matching (Chroma vector search) + LLM (Qwen).

## Cấu trúc thư mục

```
System_ChatBox/
├── .env.example          # Mẫu khai báo biến môi trường — copy thành .env rồi điền giá trị thật
├── requirements.txt
├── few_shot_examples.py  # Mẫu phong cách hội thoại cho Qwen
├── session_manager.py    # DialogueService — quyết định hành động mỗi lượt chat
├── need_extractor.py     # NeedExtractor — trích slot nhu cầu từ tin nhắn (gọi Qwen)
├── import_to_mysql.py    # Nạp catalog sản phẩm (CSV) vào MySQL
├── build_vector_index.py # Mã hóa embedding_text -> Chroma (chạy 1 lần, offline)
├── vector_search.py      # AI Matching — xếp hạng ngữ nghĩa trên Chroma
├── test_qwen_api.py      # Test kết nối Qwen API
└── main.py                # FastAPI backend chính (endpoint /chat)
```

## Lưu ý quan trọng: KHÔNG có dữ liệu (Dataset/) trong repo này

Thư mục `Dataset/` (catalog sản phẩm CSV, dữ liệu hội thoại JSON) đã bị loại khỏi repo (`.gitignore`) vì:
- Một số file vượt quá giới hạn 100MB/file của GitHub.
- Đây là dữ liệu, không phải mã nguồn, và có thể tái tạo lại bằng script.
- Một phần dữ liệu (catalog Home & Kitchen) do thành viên khác trong nhóm xử lý.

Để chạy được hệ thống, bạn cần tự có file catalog CSV (`amazon_home_kitchen_catalog_ready.csv`) và làm theo bước Setup dưới đây.

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

5. Build vector index (chạy 1 lần, hoặc khi catalog đổi):
   ```
   python build_vector_index.py
   ```

6. Test kết nối Qwen:
   ```
   python test_qwen_api.py
   ```

7. Chạy hệ thống:
   ```
   uvicorn main:app --reload
   ```
   Mở `http://127.0.0.1:8000/docs` để test qua Swagger UI.

## Kiến trúc

Xem chi tiết trong báo cáo `BaoCao_HeThongChatbot_SystemChatBox.docx` (kiến trúc RAG 2 giai đoạn: Indexing offline + Retrieval/Generation runtime, sơ đồ minh họa, mô tả từng file, kết quả thực nghiệm).

## Đề tài
Hệ thống tư vấn sản phẩm thông minh theo nhu cầu người dùng — CNTT-KLCN202.
