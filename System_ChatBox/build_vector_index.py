"""
build_vector_index.py
Mã hóa cột `embedding_text` của toàn bộ sản phẩm (đọc từ MySQL, bảng đã
tạo bởi import_to_mysql.py) thành vector, lưu vào Chroma (vector database,
persist xuống đĩa) — đây là giai đoạn INDEXING của RAG, chạy OFFLINE, MỘT
LẦN DUY NHẤT (hoặc chạy lại khi catalog sản phẩm thay đổi) — KHÔNG chạy
lại mỗi lượt chat.

Dùng sentence-transformers (model đa ngôn ngữ, chạy LOCAL trên máy bạn) để
mã hóa — KHÔNG dùng Qwen Embedding API, vì:
  - Miễn phí hoàn toàn, không tốn quota Qwen (dành quota cho sinh câu trả lời).
  - Không phụ thuộc mạng/tài khoản Qwen — encode 15.714 sản phẩm 1 lần sẽ
    tốn rất nhiều token nếu qua API, trong khi chạy local thì free.

CHUẨN BỊ TRƯỚC KHI CHẠY (không liên quan Qwen):
  1. Đã có MySQL với bảng `products` (xem import_to_mysql.py) — BẮT BUỘC.
  2. Cài thư viện (lưu ý: sentence-transformers khá nặng, cần vài phút cài
     + tải model lần đầu ~470MB):
       pip install sentence-transformers chromadb sqlalchemy pymysql python-dotenv
  3. Không cần sửa .env gì thêm — dùng chung DB_HOST/DB_PORT/... đã có.
     Có thể tuỳ chỉnh thêm (không bắt buộc):
       CHROMA_DB_PATH=./chroma_db
       EMBEDDING_MODEL_NAME=paraphrase-multilingual-MiniLM-L12-v2

CHẠY: python build_vector_index.py
KẾT QUẢ: 1 thư mục `chroma_db/` chứa toàn bộ vector đã lưu — vector_search.py
sẽ đọc thư mục này khi hệ thống chạy thật.
"""

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from sqlalchemy import create_engine, text
except ImportError:
    print("Thiếu thư viện. Chạy: pip install sqlalchemy pymysql")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Thiếu thư viện. Chạy: pip install sentence-transformers")
    sys.exit(1)

try:
    import chromadb
except ImportError:
    print("Thiếu thư viện. Chạy: pip install chromadb")
    sys.exit(1)


DB_URL = (
    f"mysql+pymysql://{os.getenv('DB_USER','root')}:{os.getenv('DB_PASSWORD','')}"
    f"@{os.getenv('DB_HOST','localhost')}:{os.getenv('DB_PORT','3306')}/{os.getenv('DB_NAME','klcn_chatbot')}"
    f"?charset=utf8mb4"
)
PRODUCTS_TABLE = "products"

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = "products"

# Model đa ngôn ngữ nhẹ (~470MB, 384 chiều) — đủ tốt để bắc cầu ngữ nghĩa
# giữa câu tiếng Việt của user và mô tả sản phẩm tiếng Anh trong catalog.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")

BATCH_SIZE = 256


def fetch_products():
    engine = create_engine(DB_URL)
    query = text(f"SELECT product_id, title, embedding_text, price, leaf_category FROM {PRODUCTS_TABLE}")
    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()
    return [dict(r) for r in rows]


def build_index():
    print(f"Đang đọc sản phẩm từ MySQL (bảng `{PRODUCTS_TABLE}`) ...")
    products = fetch_products()
    print(f"Đọc được {len(products)} sản phẩm.")

    if not products:
        print("[Lỗi] Không có sản phẩm nào — kiểm tra lại MySQL đã import chưa (import_to_mysql.py).")
        return

    print(f"Đang tải embedding model '{EMBEDDING_MODEL_NAME}' (lần đầu sẽ tải về, có thể mất vài phút) ...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"Khởi tạo Chroma persistent client tại '{CHROMA_DB_PATH}' ...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    ids, texts, metadatas = [], [], []
    for p in products:
        raw_text = p.get("embedding_text") or p.get("title") or ""
        if not raw_text.strip():
            continue  # sản phẩm không có mô tả gì để encode thì bỏ qua, không nhét chuỗi rỗng
        ids.append(str(p["product_id"]))
        texts.append(raw_text)
        metadatas.append({
            "title": (p.get("title") or "")[:500],
            "price": float(p["price"]) if p.get("price") is not None else 0.0,
            "leaf_category": (p.get("leaf_category") or "")[:200],
        })

    print(f"Đang mã hóa {len(texts)} mô tả sản phẩm thành vector (batch {BATCH_SIZE}) ...")
    total_added = 0
    for i in range(0, len(texts), BATCH_SIZE):
        batch_ids = ids[i:i + BATCH_SIZE]
        batch_texts = texts[i:i + BATCH_SIZE]
        batch_meta = metadatas[i:i + BATCH_SIZE]

        embeddings = model.encode(batch_texts, show_progress_bar=False, normalize_embeddings=True).tolist()

        collection.upsert(ids=batch_ids, embeddings=embeddings, metadatas=batch_meta, documents=batch_texts)
        total_added += len(batch_ids)
        print(f"  Đã xử lý {total_added}/{len(texts)} sản phẩm...")

    print(f"Hoàn tất. Đã lưu {total_added} vector vào Chroma collection '{COLLECTION_NAME}' tại '{CHROMA_DB_PATH}'.")
    print("Bước tiếp theo: main.py sẽ tự động dùng vector_search.py để đọc index này khi cần AI Matching.")


if __name__ == "__main__":
    build_index()