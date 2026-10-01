"""
inspect_chroma.py
Xem nhanh nội dung ĐANG LƯU THẬT trong 1 collection Chroma — dùng để kiểm
tra bằng mắt sau khi build_vector_index.py / build_category_index.py chạy
xong, KHÔNG sửa/xóa gì, chỉ đọc (read-only).

Dùng được cho CẢ 2 chỉ mục đang có trong hệ thống:
  - chroma_db/          collection "products"        (build_vector_index.py)
  - category_index_db/  collection "leaf_categories"  (build_category_index.py)

CHẠY:
  python inspect_chroma.py                                  # mặc định: xem chroma_db/products
  python inspect_chroma.py --path ./chroma_db --collection products
  python inspect_chroma.py --path ./category_index_db --collection leaf_categories
  python inspect_chroma.py --id B08XXXXXXX                  # xem đúng 1 ID cụ thể (sản phẩm)
  python inspect_chroma.py --count-only                      # chỉ đếm tổng số vector, không in mẫu
"""

import argparse

try:
    import chromadb
except ImportError:
    print("Thiếu thư viện. Chạy: pip install chromadb")
    raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Xem nhanh nội dung 1 collection Chroma.")
    parser.add_argument("--path", default="./chroma_db", help="Đường dẫn thư mục Chroma (mặc định: ./chroma_db)")
    parser.add_argument("--collection", default="products", help="Tên collection (mặc định: products)")
    parser.add_argument("--sample", type=int, default=5, help="Số mẫu in ra xem thử (mặc định: 5)")
    parser.add_argument("--id", default=None, help="Xem đúng 1 ID cụ thể thay vì lấy mẫu ngẫu nhiên")
    parser.add_argument("--count-only", action="store_true", help="Chỉ in tổng số vector, không in mẫu")
    args = parser.parse_args()

    try:
        client = chromadb.PersistentClient(path=args.path)
        collection = client.get_collection(name=args.collection)
    except Exception as e:
        print(f"[LỖI] Không mở được collection '{args.collection}' tại '{args.path}': {e}")
        print("=> Kiểm tra lại --path/--collection, hoặc đã chạy build_vector_index.py / "
              "build_category_index.py chưa.")
        return

    total = collection.count()
    print(f"Collection: '{args.collection}'  |  Đường dẫn: '{args.path}'")
    print(f"Tổng số vector đang lưu: {total}")

    if args.count_only:
        return

    if args.id:
        result = collection.get(ids=[args.id], include=["embeddings", "metadatas", "documents"])
        if not result["ids"]:
            print(f"\n[Không tìm thấy] ID '{args.id}' không có trong collection.")
            return
        print(f"\n=== Chi tiết ID '{args.id}' ===")
        print("Document (văn bản đã encode):", result["documents"][0])
        print("Metadata:", result["metadatas"][0])
        vec = result["embeddings"][0]
        print(f"Vector: {len(vec)} chiều, 5 giá trị đầu: {[round(float(v), 4) for v in vec[:5]]}...")
        return

    # Lấy mẫu đầu tiên (Chroma không có "random sample" built-in đơn giản,
    # nên đây là N phần tử ĐẦU theo thứ tự lưu trữ nội bộ, không phải ngẫu
    # nhiên thật sự — đủ dùng để kiểm tra bằng mắt).
    result = collection.get(limit=args.sample, include=["embeddings", "metadatas", "documents"])
    print(f"\n=== {len(result['ids'])} mẫu đầu tiên ===")
    for i, doc_id in enumerate(result["ids"]):
        doc = result["documents"][i] if result["documents"] else None
        meta = result["metadatas"][i] if result["metadatas"] else None
        vec = result["embeddings"][i] if result["embeddings"] is not None else None
        print(f"\n--- [{i+1}] ID: {doc_id} ---")
        if doc:
            preview = doc[:150] + ("..." if len(doc) > 150 else "")
            print(f"  Document: {preview}")
        if meta:
            print(f"  Metadata: {meta}")
        if vec is not None:
            print(f"  Vector: {len(vec)} chiều, 5 giá trị đầu: {[round(float(v), 4) for v in vec[:5]]}...")


if __name__ == "__main__":
    main()