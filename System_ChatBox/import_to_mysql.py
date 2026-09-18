"""
import_to_mysql.py
Nạp `amazon_home_kitchen_catalog_ready.csv` (Bộ 1, 15.714 sản phẩm, 23 cột)
vào MySQL.

Khác với cách làm "hard-code tên cột" thông thường: script này dùng
pandas.to_sql để TỰ TẠO bảng `products` theo ĐÚNG cột thật có trong CSV
(vì CSV do người phụ trách Bộ 1 xuất ra, cột có thể khác giữa các lần xử
lý) — sau đó mới ALTER TABLE thêm khóa chính `product_id` + index cho
`price`/`leaf_category`/`main_category`/`brand` bằng SQL thuần.

Ngoài bảng `products`, script tạo thêm 2 bảng phụ dùng chung cho toàn hệ
thống:
  - conversation_sessions: log lịch sử phiên chat (có thể đồng bộ với
    SessionState trong session_manager.py khi cần lưu lâu dài).
  - evaluation_logs: log Precision@K + response time (Chức năng 4 —
    đánh giá hệ thống theo đề cương).

CHUẨN BỊ TRƯỚC KHI CHẠY (không liên quan Qwen, chỉ cần MySQL):
  1. Cài MySQL Server, tạo database:
       CREATE DATABASE klcn_chatbot CHARACTER SET utf8mb4;
  2. Cài thư viện:
       pip install pandas sqlalchemy pymysql python-dotenv
  3. Tạo file .env cùng thư mục System_ChatBox/:
       DB_HOST=localhost
       DB_PORT=3306
       DB_USER=root
       DB_PASSWORD=your_password
       DB_NAME=klcn_chatbot
  4. Sửa CATALOG_CSV_PATH bên dưới trỏ đúng file catalog_ready.csv thật.

File này KHÔNG cần Qwen API key.
"""

import json
import os
import sys

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from sqlalchemy import create_engine, text
except ImportError:
    print("Thiếu thư viện. Chạy: pip install sqlalchemy pymysql pandas python-dotenv")
    sys.exit(1)


# ---------- Cấu hình ----------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "klcn_chatbot")

DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

# Sửa đường dẫn này cho đúng file catalog_ready.csv thật của Bộ 1
CATALOG_CSV_PATH = os.getenv(
    "CATALOG_CSV_PATH",
    "../Dataset/Dataset1 - Home & Kitchen 40k/Data_Cleaning/amazon_home_kitchen_catalog_ready.csv",
)

PRODUCTS_TABLE = "products"
PRIMARY_KEY_COL = "product_id"   # khóa chính thật (theo CSV thật: product_id, không phải parent_asin)

# Các cột thường dùng để lọc rule-based / xếp hạng — sẽ đánh index nếu
# tồn tại trong CSV thật (kiểm tra động, không giả định cột nào chắc chắn có)
INDEX_CANDIDATE_COLS = ["price", "leaf_category", "main_category", "brand"]

CREATE_SESSIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS conversation_sessions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL UNIQUE,
    turn_count INT DEFAULT 0,
    slots_json JSON,
    ended TINYINT(1) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

CREATE_EVAL_LOGS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS evaluation_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(64),
    turn_index INT,
    precision_at_k DECIMAL(5,4),
    k_value INT,
    response_time_ms INT,
    product_relevance_score DECIMAL(5,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""


def get_engine():
    try:
        engine = create_engine(DB_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        print(f"[Lỗi kết nối MySQL] {e}")
        print("=> Kiểm tra: MySQL đã chạy chưa? .env đúng thông tin chưa? Database đã tạo chưa?")
        sys.exit(1)


def ensure_auxiliary_tables(engine):
    with engine.begin() as conn:
        conn.execute(text(CREATE_SESSIONS_TABLE_SQL))
        conn.execute(text(CREATE_EVAL_LOGS_TABLE_SQL))
    print("Đã đảm bảo 2 bảng phụ: conversation_sessions, evaluation_logs.")


def import_catalog(csv_path: str, if_exists: str = "replace", chunksize: int = 500):
    if not os.path.exists(csv_path):
        print(f"[Lỗi] Không tìm thấy file: {csv_path}")
        print("=> Sửa CATALOG_CSV_PATH cho đúng đường dẫn catalog_ready.csv thật.")
        return

    print(f"Đang đọc {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"Đọc được {len(df)} dòng, {len(df.columns)} cột.")
    print(f"Các cột: {list(df.columns)}")

    if PRIMARY_KEY_COL not in df.columns:
        print(f"[Cảnh báo] Không thấy cột '{PRIMARY_KEY_COL}' trong CSV — "
              f"bảng sẽ được tạo KHÔNG có khóa chính tự nhiên, chỉ có id tự tăng.")

    # details_json / embedding_text nếu là dict/list thì ép về string JSON
    # để MySQL lưu được (cột JSON hoặc TEXT tuỳ pandas suy luận kiểu).
    # Dùng json.dumps chuẩn của Python (KHÔNG dùng pd.io.json.ujson_dumps —
    # API nội bộ này đã bị gỡ bỏ ở pandas 2.x, dễ crash nếu cột thực sự
    # chứa object dict/list thay vì chuỗi text như CSV thường có).
    for col in df.columns:
        if df[col].apply(lambda v: isinstance(v, (dict, list))).any():
            df[col] = df[col].apply(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)

    engine = get_engine()

    print(f"Đang tạo/ghi bảng `{PRODUCTS_TABLE}` (if_exists='{if_exists}') ...")
    df.to_sql(PRODUCTS_TABLE, con=engine, if_exists=if_exists, index=False, chunksize=chunksize, method="multi")
    print(f"Đã import {len(df)} sản phẩm vào bảng `{PRODUCTS_TABLE}`.")

    # Thêm khóa chính + index sau khi bảng đã tồn tại (to_sql không tự set PK)
    with engine.begin() as conn:
        if PRIMARY_KEY_COL in df.columns:
            try:
                conn.execute(text(
                    f"ALTER TABLE {PRODUCTS_TABLE} "
                    f"MODIFY {PRIMARY_KEY_COL} VARCHAR(64) NOT NULL, "
                    f"ADD PRIMARY KEY ({PRIMARY_KEY_COL})"
                ))
                print(f"Đã set khóa chính: {PRIMARY_KEY_COL}")
            except Exception as e:
                print(f"[Cảnh báo] Không set được khóa chính {PRIMARY_KEY_COL}: {e}")
                print("  (có thể do trùng giá trị hoặc cột chứa NULL — kiểm tra lại dữ liệu gốc)")

        for col in INDEX_CANDIDATE_COLS:
            if col in df.columns:
                # Cột kiểu chữ mà pandas.to_sql suy luận thành TEXT trong MySQL
                # cần chỉ định độ dài index (prefix length) — 191 ký tự là mức
                # an toàn chuẩn cho utf8mb4 (giới hạn khóa 767 byte / 4 byte mỗi
                # ký tự). Dùng is_numeric_dtype thay vì so dtype == "object" vì
                # pandas bản mới (2.x/3.x) có thể đọc cột chữ thành kiểu
                # "string" (StringDtype) thay vì "object" như bản cũ.
                is_text_col = not pd.api.types.is_numeric_dtype(df[col])
                index_expr = f"{col}(191)" if is_text_col else col
                try:
                    conn.execute(text(f"CREATE INDEX idx_{col} ON {PRODUCTS_TABLE} ({index_expr})"))
                    print(f"Đã tạo index cho cột: {col}")
                except Exception as e:
                    print(f"[Bỏ qua index {col}]: {e}")

    ensure_auxiliary_tables(engine)
    print("Hoàn tất import.")


if __name__ == "__main__":
    import_catalog(CATALOG_CSV_PATH)
