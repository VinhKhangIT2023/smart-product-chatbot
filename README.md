# 🛍️ Smart Product Advisor Chatbot — CNTT-KLCN202

> Hệ thống tư vấn sản phẩm thông minh theo nhu cầu người dùng — chatbot hội thoại nhiều lượt, kiến trúc RAG kết hợp **Rule-based** (MySQL) + **AI Matching** (vector embedding, ChromaDB) + **LLM** (Qwen).

Đồ án khóa luận cử nhân CNTT, Khoa Công nghệ Thông tin — Trường Đại học Công Thương TP.HCM (mã đề tài `CNTT-KLCN202`, năm học 2026–2027).

---

## ✨ Tính năng chính

- **Hội thoại nhiều lượt có ngữ cảnh** — thu thập dần nhu cầu người dùng (loại sản phẩm, kích thước, giá, màu, thương hiệu...) qua nhiều lượt chat, không hỏi dồn một lúc.
- **Lọc cứng theo quy tắc (Rule-based)** trên MySQL: category, khoảng giá, màu sắc (hỗ trợ sản phẩm nhiều màu qua `JSON_CONTAINS`), thương hiệu, kích thước sản phẩm (dung sai có thể cấu hình).
- **Phân biệt ràng buộc bắt buộc vs. ưu tiên mềm** — "bắt buộc màu đỏ" sẽ loại sản phẩm không đúng màu; "ưu tiên màu đỏ" chỉ ảnh hưởng xếp hạng, không loại sản phẩm thiếu dữ liệu.
- **AI Matching ngữ nghĩa (Tầng 2)** — xếp hạng lại tập ứng viên bằng cosine similarity trên vector embedding đa ngôn ngữ (`ChromaDB`), không cần gọi thêm LLM.
- **Dịch thuộc tính tiếng Việt → dữ liệu catalog gốc tiếng Anh** — category (vector similarity + bảng gợi ý tay cho từ mơ hồ), màu sắc (bảng ánh xạ + taxonomy dùng chung với pipeline làm sạch dữ liệu), tiền tệ (VND ↔ USD theo tỷ giá cố định, minh bạch với người dùng).
- **Gợi ý sản phẩm "gần nhất"** khi không có kết quả khớp tuyệt đối — nới từng tiêu chí một, ghi rõ sản phẩm lệch ở đâu, không bao giờ ngụ ý sản phẩm chưa khớp là đã khớp.
- **Sinh phản hồi tự nhiên bằng LLM (Qwen)** — tách biệt hoàn toàn 2 vai trò: trích xuất nhu cầu (NeedExtractor) và sinh câu trả lời (ResponseService), đúng 2 lần gọi LLM/lượt chat, chỉ tư vấn dựa trên sản phẩm thật có trong hệ thống.

## 🏗️ Kiến trúc

```
User
 │
 ▼
need_extractor.py ──(Qwen #1: trích JSON nhu cầu)──▶ session_manager.py
                                                            │ (đủ thông tin?)
                                                            ▼
                                          ┌── rule_based_filter() [MySQL — Tầng 1: lọc cứng]
                                          │        category / giá / màu / brand / kích thước
                                          ▼
                                   vector_search.py [ChromaDB — Tầng 2: AI Matching]
                                          │        xếp hạng Top-K theo ngữ nghĩa
                                          ▼
                              (0 kết quả?) find_near_matches() [nới từng tiêu chí]
                                          │
                                          ▼
                         generate_llm_reply() ──(Qwen #2: sinh câu trả lời tự nhiên)──▶ User
```

## 🧰 Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| Backend | Python, FastAPI, Uvicorn |
| Cơ sở dữ liệu có cấu trúc | MySQL (SQLAlchemy + PyMySQL) |
| Vector Database | ChromaDB (lưu trên đĩa, persistent) |
| Embedding model | `paraphrase-multilingual-MiniLM-L12-v2` (Sentence-Transformers) |
| LLM | Qwen (qua OpenRouter, giao thức OpenAI-compatible) |
| Khác | pandas, python-dotenv |

## 📁 Cấu trúc thư mục

```
System_ChatBox/
├── main.py                     # FastAPI app — điểm khởi chạy (entrypoint)
│
├── src/                        # Mã nguồn lõi — chạy MỖI LƯỢT CHAT thật
│   ├── session_manager.py      #   DialogueService: quản lý phiên, quyết định hành động
│   ├── need_extractor.py       #   NeedExtractor: trích slot nhu cầu (gọi Qwen #1)
│   ├── few_shot_examples.py    #   Mẫu phong cách hội thoại cho ResponseService
│   ├── vector_search.py        #   AI Matching: xếp hạng ngữ nghĩa (Tầng 2)
│   ├── category_mapper.py      #   Dịch category VI → leaf_category EN thật
│   ├── color_mapper.py         #   Dịch màu VI → màu chuẩn EN
│   ├── color_normalization.py  #   Taxonomy màu dùng chung (cleaning + runtime)
│   ├── currency.py             #   Quy đổi VND ↔ USD (tỷ giá cố định)
│   └── size_normalization.py   #   Phân tích kích thước sản phẩm từ text tự do
│
├── scripts/                    # Chạy OFFLINE — build/nạp dữ liệu, không chạy mỗi lượt chat
│   ├── import_to_mysql.py      #   Nạp catalog CSV vào MySQL
│   ├── build_vector_index.py   #   Encode sản phẩm → ChromaDB (chroma_db/)
│   └── build_category_index.py #   Encode tên category → ChromaDB (category_index_db/)
│
├── tools/                      # Công cụ dev/chẩn đoán — không thuộc luồng chat thật
│   ├── inspect_chroma.py       #   Xem trực tiếp nội dung đã lưu trong ChromaDB
│   ├── test_qwen_api.py        #   Kiểm tra kết nối Qwen API
│   └── batch_test_category_mapper.py  # Kiểm tra hàng loạt chất lượng dịch category
│
├── chroma_db/                  # (tự sinh, gitignore) vector sản phẩm
├── category_index_db/          # (tự sinh, gitignore) vector tên category
├── venv/                       # (gitignore) môi trường ảo Python
├── .env / .env.example
├── requirements.txt
└── .gitignore
```

## 🚀 Bắt đầu

### Yêu cầu

- Python 3.11+ (khuyến nghị — một số bản Python rất mới như 3.14 có thể gặp xung đột với `torch`/`sentence-transformers`)
- MySQL Server đang chạy
- API key Qwen (qua [OpenRouter](https://openrouter.ai) hoặc Alibaba Cloud)

### Cài đặt

```bash
# 1. Clone repo
git clone <repo-url>
cd System_ChatBox

# 2. Tạo và kích hoạt môi trường ảo
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 3. Cài thư viện
pip install -r requirements.txt
```

### Cấu hình

```bash
cp .env.example .env
```

Điền vào `.env`:

```dotenv
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_mysql_password
DB_NAME=klcn_chatbot

QWEN_API_KEY=your_openrouter_or_alibaba_api_key
QWEN_BASE_URL=https://openrouter.ai/api/v1
QWEN_MODEL=qwen/qwen3.8-flash

CATALOG_CSV_PATH=duong_dan_toi_file_catalog.csv   # BẮT BUỘC, không có giá trị mặc định
```

> ⚠️ **Luôn chạy mọi lệnh từ thư mục gốc `System_ChatBox/`**, không `cd` vào `scripts/`/`tools/` trước — các đường dẫn tương đối trong `.env` (`CHROMA_DB_PATH`...) được tính theo thư mục đang đứng khi chạy lệnh, `cd` sai sẽ khiến chương trình tìm nhầm chỗ.

### Khởi tạo dữ liệu (chạy 1 lần, theo đúng thứ tự)

```bash
# Tạo database rỗng trước (1 lần)
mysql -u root -p -e "CREATE DATABASE klcn_chatbot CHARACTER SET utf8mb4;"

# Nạp catalog sản phẩm vào MySQL
python scripts/import_to_mysql.py

# Build chỉ mục vector cho sản phẩm (AI Matching)
python scripts/build_vector_index.py

# Build chỉ mục vector cho tên category (dịch category VI -> EN)
python scripts/build_category_index.py
```

### Chạy thử

```bash
# Kiểm tra kết nối Qwen
python tools/test_qwen_api.py

# Khởi động server
uvicorn main:app --reload
```

Mở **http://127.0.0.1:8000/docs** để test qua Swagger UI.

## 🔄 Khi cập nhật dataset mới

Không cần "train lại" gì cả — hệ thống không huấn luyện model nào, chỉ **encode** (suy luận) bằng model đã có sẵn:

| Thay đổi | Cần chạy lại |
|---|---|
| Thêm/sửa dữ liệu MySQL (giá, brand...) | Không cần gì — `rule_based_filter()` đọc trực tiếp MySQL mỗi lượt chat |
| `embedding_text` đổi (mô tả, màu...) | `python scripts/build_vector_index.py` |
| `leaf_category` có giá trị mới | `python scripts/build_category_index.py` |

## ⚠️ Hạn chế đã biết

- **Chưa có giao diện web** — hiện chỉ có backend API (FastAPI), test qua Swagger UI.
- **Lịch sử hội thoại chỉ lưu trong bộ nhớ** (in-memory), mất khi restart server — bảng `conversation_sessions` đã tạo sẵn trong MySQL nhưng chưa có code ghi vào.
- **Chưa đo Precision@K / Product Relevance Score** — bảng `evaluation_logs` đã tạo sẵn nhưng chưa có pipeline đánh giá tự động.
- **Dữ liệu màu còn một phần chưa xác định được** (`color_status = needs_review/missing`) — do dữ liệu nguồn Amazon không đầy đủ, không phải lỗi logic.
- **`VI_CATEGORY_HINTS`** (bảng gợi ý category cho từ tiếng Việt ngắn/mơ hồ) mới phủ ~65 từ khóa phổ biến, chưa toàn diện.

## 👥 Nhóm thực hiện

Khóa luận cử nhân, Khoa Công nghệ Thông tin, Trường Đại học Công Thương TP.HCM — GVHD: ThS. Nguyễn Thị Định.