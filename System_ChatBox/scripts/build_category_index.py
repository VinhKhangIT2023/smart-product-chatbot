"""
build_category_index.py
Script OFFLINE — chạy 1 lần (giống tinh thần build_vector_index.py cho sản phẩm),
KHÔNG chạy mỗi request. Giải quyết vấn đề đã phát hiện:

  need_extractor.py trích xuất category bằng TIẾNG VIỆT (vd "thảm trải sàn"),
  nhưng cột leaf_category/main_category trong bảng `products` (MySQL) là
  TIẾNG ANH nguyên gốc từ Amazon (vd "Area Rugs") -> rule_based_filter() dùng
  LIKE '%thảm trải sàn%' không bao giờ khớp -> luôn trả 0 candidates.

  Đã xác nhận thêm: main_category CHỈ CÓ ĐÚNG 1 GIÁ TRỊ DUY NHẤT
  "Home_and_Kitchen" cho toàn bộ 15.714 dòng (kiểm tra qua DISTINCT thật) ->
  lọc theo main_category vô nghĩa, chỉ leaf_category mới có tác dụng phân loại.

Cách giải quyết: KHÔNG dịch tay 800+ leaf_category (dễ thiếu/sai khi data đổi),
mà xây 1 CHỈ MỤC VECTOR RIÊNG, NHỎ, chỉ chứa tên các leaf_category — dùng
CHUNG model embedding đa ngôn ngữ đã dùng cho sản phẩm
(paraphrase-multilingual-MiniLM-L12-v2, đã encode tốt cả câu tiếng Việt lẫn
tên category tiếng Anh vào cùng không gian ngữ nghĩa). Runtime, category_mapper.py
sẽ encode câu tiếng Việt user nói và tìm các leaf_category tiếng Anh gần nghĩa
nhất trong chỉ mục nhỏ này (rất nhanh, vài trăm vector) -> dùng làm điều kiện
`leaf_category IN (...)` cho rule_based_filter(), KHÔNG cần gọi thêm Qwen.

CHẠY:
  python build_category_index.py
Cần đã có .env với DB_HOST/DB_PORT/DB_USER/DB_PASSWORD/DB_NAME (giống main.py)
và đã cài sentence-transformers, chromadb (đã có trong requirements.txt).
"""

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import chromadb

DB_URL = (
    f"mysql+pymysql://{os.getenv('DB_USER','root')}:{os.getenv('DB_PASSWORD','')}"
    f"@{os.getenv('DB_HOST','localhost')}:{os.getenv('DB_PORT','3306')}/{os.getenv('DB_NAME','klcn_chatbot')}"
    f"?charset=utf8mb4"
)
PRODUCTS_TABLE = "products"

# Đặt riêng 1 path khác với chroma_db của sản phẩm (./chroma_db) để không lẫn
# 2 collection có mục đích khác nhau (sản phẩm vs tên category) vào cùng chỗ.
CATEGORY_DB_PATH = os.getenv("CATEGORY_DB_PATH", "./category_index_db")
COLLECTION_NAME = "leaf_categories"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")


def fetch_distinct_leaf_categories() -> list:
    engine = create_engine(DB_URL)
    query = text(
        f"SELECT DISTINCT leaf_category FROM {PRODUCTS_TABLE} "
        f"WHERE leaf_category IS NOT NULL AND leaf_category <> ''"
    )
    with engine.connect() as conn:
        rows = conn.execute(query).fetchall()
    # loại khoảng trắng thừa, loại trùng lặp phát sinh sau khi strip
    values = sorted({r[0].strip() for r in rows if r[0] and r[0].strip()})
    return values


def main():
    print("[build_category_index] Đang lấy danh sách leaf_category từ MySQL...")
    try:
        categories = fetch_distinct_leaf_categories()
    except Exception as e:
        print(f"[build_category_index] LỖI kết nối/truy vấn MySQL: {e}")
        sys.exit(1)

    if not categories:
        print("[build_category_index] Không tìm thấy leaf_category nào — kiểm tra lại bảng products.")
        sys.exit(1)

    print(f"[build_category_index] Tìm thấy {len(categories)} leaf_category duy nhất.")

    print(f"[build_category_index] Đang nạp model embedding ({EMBEDDING_MODEL_NAME})...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("[build_category_index] Đang encode...")
    embeddings = model.encode(categories, normalize_embeddings=True, show_progress_bar=True)

    print(f"[build_category_index] Đang ghi vào Chroma tại {CATEGORY_DB_PATH} ...")
    client = chromadb.PersistentClient(path=CATEGORY_DB_PATH)
    # Xoá collection cũ nếu đã tồn tại, để build lại sạch (category có thể đổi
    # theo thời gian nếu quản trị viên thêm/sửa sản phẩm — UC-12 trong khoá luận).
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass  # chưa có collection cũ thì thôi

    collection = client.create_collection(name=COLLECTION_NAME)

    # Dùng chính tên category (đã strip) làm id — duy nhất, dễ đọc khi debug.
    collection.add(
        ids=categories,
        embeddings=[e.tolist() for e in embeddings],
        metadatas=[{"leaf_category": c} for c in categories],
        documents=categories,
    )

    print(f"[build_category_index] XONG. Đã lưu {len(categories)} vector category vào "
          f"collection '{COLLECTION_NAME}' tại {CATEGORY_DB_PATH}.")
    print("[build_category_index] Chạy lại script này bất cứ khi nào danh mục sản phẩm "
          "trong MySQL thay đổi (thêm category mới) để chỉ mục luôn khớp dữ liệu thật.")


if __name__ == "__main__":
    main()