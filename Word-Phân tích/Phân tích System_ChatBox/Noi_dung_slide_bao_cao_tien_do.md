# Nội dung slide báo cáo tiến độ KLCN202

## Slide 1 — Trang bìa

Tiêu đề: BÁO CÁO TIẾN ĐỘ KHÓA LUẬN.

Tên đề tài và mã đề tài bên dưới.

Dòng phụ: “Xây dựng backend chatbot tư vấn sản phẩm Home & Kitchen”.

Có vùng điền nhóm ba thành viên, giảng viên hướng dẫn và ngày báo cáo.

Hình minh họa: chỉ dùng biểu tượng nhỏ sản phẩm, hội thoại và cơ sở dữ liệu; không dùng ảnh trang trí lớn.

## Slide 2 — Phạm vi công việc và tiến độ nổi bật

Thông điệp: đã chuẩn bị được dữ liệu phục vụ backend và chỉ mục vector; luồng chat cần tiếp tục kiểm chứng.

Trình bày ba nhóm công việc:

- Tiếp nhận catalog Bộ 1 đã xử lý và nạp vào MySQL.
- Chuẩn bị ví dụ hội thoại, quản lý phiên và các module tư vấn.
- Tích hợp Qwen và kiểm thử toàn luồng: chưa hoàn tất theo báo cáo.

Hiển thị hai số liệu nổi bật:

- 15.714 sản phẩm đã import.
- 15.714 vector đã lưu vào Chroma.

Chú thích: đây là kết quả của phần chuẩn bị dữ liệu, không phải số lượt tư vấn thành công.

Nguồn nội bộ: mục 1, mục 6.1 và Phụ lục A của báo cáo.

## Slide 3 — Kiến trúc tổng thể và vai trò các thành phần

Chia slide thành hai phần:

- Offline: CSV → MySQL → mô hình embedding → Chroma.
- Runtime: tin nhắn → trích nhu cầu → quyết định hành động → lọc/xếp hạng khi cần → sinh phản hồi.

Bên dưới có ba thẻ vai trò:

- Rule-based: lọc theo điều kiện được biểu diễn bằng quy tắc và truy vấn SQL.
- AI Matching: xếp hạng ngữ nghĩa trong tập ứng viên.
- Qwen: trích xuất nhu cầu và tạo câu trả lời tự nhiên.

Ghi rõ: “Sử dụng mô hình có sẵn; không huấn luyện mô hình mới”.

Đây là sơ đồ tóm tắt, không chèn cả hai hình chi tiết vào cùng slide.

## Slide 4 — Luồng xử lý dữ liệu offline

Dùng nguyên Hình 1 đính kèm làm hình chính, đặt lớn ở giữa, giữ đúng tỉ lệ.

Không vẽ lại thành một luồng khác và không cắt mất tên bước hoặc mũi tên.

Chỉ thêm một chú thích ngắn bên dưới:

“Thực hiện khi khởi tạo hoặc cập nhật catalog; không chạy lại trong mỗi lượt chat.”

Ghi chú thuyết trình giải thích năm bước:

1. Nhận CSV catalog.
2. Nạp dữ liệu.
3. Lưu dữ liệu có cấu trúc trong MySQL.
4. Tạo embedding bằng mô hình chạy local.
5. Lưu vector vào Chroma.

## Slide 5 — Kết quả nạp và kiểm tra dữ liệu MySQL

Dùng thẻ số liệu kết hợp bảng ngắn:

| Nội dung kiểm tra | Kết quả theo báo cáo |
|---|---|
| Số dòng import | 15.714/15.714 |
| Số cột | 23 |
| Khóa chính | product_id |
| Index bổ sung | 4: price, leaf_category, main_category, brand |
| Số dòng có price NULL hoặc bằng 0 | 0 |
| Số giá trị brand phân biệt | 9.347 |

Ghi nguồn: “Báo cáo, mục 4.3; Phụ lục A.3–A.4”.

Không kết luận “dữ liệu hoàn toàn sạch” chỉ từ phép kiểm tra giá. Không diễn giải 9.347 giá trị brand thành 9.347 thương hiệu đã xác thực.

Không cần liệt kê đủ 23 tên cột trên slide.

## Slide 6 — Kết quả tạo chỉ mục vector

Trình bày:

- Mô hình: paraphrase-multilingual-MiniLM-L12-v2.
- Chạy local bằng SentenceTransformer.
- Kích thước vector: 384 chiều.
- Batch size: 256 sản phẩm/lần.
- Vector đã lưu: 15.714/15.714.
- Nơi lưu: chroma_db/, collection products.

Dùng minh họa “mô tả sản phẩm → vector → Chroma”.

Có thể hiển thị thanh hoàn tất 100%, nhưng phải ghi đúng nhãn: “Hoàn tất tạo và lưu vector cho catalog hiện tại”.

Không dùng nhãn “độ chính xác 100%”.

Ghi nguồn: mục 4.4 và Phụ lục A.2.

## Slide 7 — Luồng xử lý một lượt chat

Dùng nguyên Hình 2 đính kèm làm hình chính. Vì hình có dạng dọc, bố trí hình ở bên trái hoặc giữa với chiều cao tối đa có thể; bên phải chỉ có ba ý ngắn:

- Qwen lần 1: trích xuất nhu cầu.
- Bộ quản lý phiên: quyết định hỏi tiếp, tìm kiếm hoặc kết thúc.
- Qwen lần 2: sinh phản hồi theo hành động và kết quả tìm kiếm nếu có.

Ghi rõ: “Luồng thiết kế; chưa kiểm chứng end-to-end tại thời điểm báo cáo”.

Diễn đạt hai lần gọi Qwen là cấu trúc của luồng xử lý dự kiến khi đi qua đầy đủ các bước, không phải thống kê từ các lượt chat chạy thành công.

Nếu chữ trong hình quá nhỏ, tạo bố cục nhiều khung phóng to từ chính hình gốc, giữ nguyên thứ tự, nhãn và các nhánh; không làm mất thông tin.

Biểu tượng website cuối luồng không được dùng làm bằng chứng frontend đã xây dựng xong.

## Slide 8 — Quản lý nhu cầu và điều phối hội thoại

Trình bày hai nhóm thông tin.

Nhóm thuộc tính nhu cầu:

- category: loại sản phẩm, bắt buộc theo thiết kế.
- size_space: kích thước/không gian; thiết kế có bước hỏi nhưng không ép người dùng cung cấp số đo chính xác.
- price_range: ngân sách, ưu tiên hỏi trong nhóm thuộc tính mềm.
- color, material, style, brand: các thuộc tính bổ sung.

Nhóm hành động:

- ASK_HARD_SLOT: hỏi thông tin cần thiết.
- ASK_SOFT_SLOT: hỏi thêm sở thích.
- SEARCH_PRODUCTS: chuyển sang tìm sản phẩm.
- END_SESSION: kết thúc phiên.

Ghi chú: giới hạn hỏi tối đa hai thuộc tính mềm mỗi phiên là lựa chọn thiết kế hiện tại, chưa trình bày như một kết luận đã chứng minh bằng thực nghiệm.

Không khẳng định hệ thống đã kiểm tra chính xác độ vừa vặn của kích thước sản phẩm.

## Slide 9 — Kết quả demo quản lý phiên

Dùng timeline ba lượt dựa đúng log trong Phụ lục A.1:

- Lượt 1: ASK_HARD_SLOT → size_space.
- Lượt 2: ASK_SOFT_SLOT → price_range.
- Lượt 3: ASK_SOFT_SLOT → color.

Hiển thị trạng thái cuối:

- category = thảm trải sàn.
- size_space = 2m × 3m.
- Các thuộc tính còn lại chưa có giá trị.

Ghi rõ “Demo riêng module session_manager.py; không phải cuộc hội thoại end-to-end với Qwen”.

Không tự dựng tin nhắn người dùng rồi trình bày như log thật. Có thể dùng thẻ trạng thái thay cho bong bóng chat.

## Slide 10 — Ví dụ hội thoại few-shot

Số liệu:

- 14 cặp ví dụ viết tay.
- Tương ứng 28 messages.
- Năm nhóm hành vi: hỏi kích thước/không gian; hỏi thuộc tính mềm; tinh chỉnh theo phản hồi; giải thích; xử lý câu hỏi mơ hồ/lạc đề/kết thúc.

Dùng một cặp ví dụ trong Phụ lục A.5:

Người dùng: “Tôi cần mua thảm trải phòng khách”.

Trợ lý mẫu: hỏi diện tích phòng khách hoặc kích thước khu vực muốn trải thảm để tìm kích thước phù hợp.

Nếu rút gọn câu trả lời, ghi “Ví dụ rút gọn từ báo cáo”.

Gắn nhãn “Ví dụ viết tay đưa vào prompt”, không gọi đây là phản hồi Qwen đã chạy thành công hoặc dữ liệu huấn luyện mô hình.

Không cần đưa ước lượng token và nhận định chi phí lên slide.

## Slide 11 — Lọc sản phẩm và xếp hạng ngữ nghĩa

Minh họa một luồng ngang:

Nhu cầu đã trích xuất → lọc SQL → tập ứng viên → lấy vector theo product_id → cosine similarity → Top-K.

Nêu ba ý:

- SQL lọc ứng viên theo các điều kiện đã triển khai.
- AI Matching lấy vector của ứng viên trong Chroma để xếp hạng.
- K mặc định bằng 5; số sản phẩm thực tế trả về có thể ít hơn nếu không đủ ứng viên.

Có chú thích nhỏ về fallback: khi không dùng được chỉ mục/vector, code có nhánh trả theo thứ tự ban đầu; không coi đây là kết quả AI Matching đã kiểm chứng.

Gắn trạng thái: “Đã viết module; chưa có kết quả kiểm thử runtime được xác nhận trong báo cáo”.

Không tự tạo sản phẩm, điểm similarity hoặc Precision@5.

## Slide 12 — Bảng tiến độ các module

Dùng bảng ba cột, không dùng biểu đồ phần trăm hoàn thành dự án:

| Module | Trạng thái theo báo cáo | Bằng chứng/phần còn thiếu |
|---|---|---|
| few_shot_examples.py | Đã chạy kiểm tra | 14 cặp ví dụ |
| session_manager.py | Đã chạy demo | Ba lượt chuyển hành động |
| import_to_mysql.py | Đã chạy | 15.714 dòng và truy vấn kiểm tra |
| build_vector_index.py | Đã chạy | Lưu 15.714 vector |
| vector_search.py | Đã viết, mới kiểm tra cú pháp | Chưa kiểm chứng runtime |
| need_extractor.py | Đã viết, mới kiểm tra cú pháp | Chờ kiểm thử với API |
| main.py | Đã viết, mới kiểm tra cú pháp | Chưa chạy end-to-end |
| test_qwen_api.py | Chưa thành công | Kết nối dịch vụ chưa thông suốt |

Phân biệt màu xanh, cam và xám, kèm chữ trạng thái để không phụ thuộc riêng vào màu sắc.

## Slide 13 — Khó khăn đã gặp và hướng xử lý

Bố trí ba hàng “Vấn đề → Xử lý/trạng thái”.

1. Index trên cột TEXT trong MySQL:
   Đã gặp lỗi tạo index; đã điều chỉnh prefix length cho cột chữ và chạy lại thành công.

2. Giả định tên cột không khớp dữ liệu:
   Đã cập nhật sang product_id, main_category và leaf_category theo CSV thật.

3. Qwen API:
   Báo cáo ghi nhận lỗi kiểm soát tài khoản ở Alibaba và lỗi model không khả dụng ở OpenRouter tại thời điểm thử; chưa xác nhận kết nối thành công.

Phần Qwen chỉ ghi hướng tiếp theo: kiểm tra tài khoản, endpoint, model và quyền sử dụng; chạy script kiểm tra API trước khi tích hợp.

Không khẳng định nguyên nhân lỗi tài khoản đã được xác định chắc chắn. Không ghi nhóm đã nạp tiền, đã mua dịch vụ hoặc API đã ổn định nếu tài liệu chưa có bằng chứng.

## Slide 14 — Kế hoạch tiếp theo và kết luận tiến độ

Trình bày danh sách có thứ tự:

1. Kiểm thử riêng module xếp hạng với MySQL và Chroma đã có.
2. Xác nhận gọi Qwen API thành công.
3. Kiểm thử trích xuất nhu cầu và luồng POST /chat.
4. Bổ sung ghi log vào conversation_sessions và evaluation_logs.
5. Chuẩn bị bộ ca đánh giá với nhãn sản phẩm phù hợp để đo Precision@K và thời gian phản hồi.
6. Xây dựng, kết nối giao diện website và kiểm thử toàn hệ thống.

Không tự gán thời hạn hoặc thành viên phụ trách.

Kết luận ngắn:

“Đã hoàn tất nạp catalog và tạo chỉ mục vector; đã xây dựng các module backend cốt lõi. Trọng tâm tiếp theo là kiểm chứng luồng chat, tích hợp giao diện và đo chất lượng tư vấn.”

Phần đo lường ghi “Dự kiến đánh giá”, không đưa số liệu giả. Không dùng các ví dụ few-shot làm bằng chứng về độ chính xác tư vấn.

## 4. Yêu cầu về hình ảnh và nguồn

Hình 1 dùng cho slide 4; Hình 2 dùng cho slide 7. Đây là hai hình chính cần giữ đúng nội dung.

Các slide còn lại dùng bảng, thẻ số liệu và sơ đồ đơn giản. Nếu tạo minh họa giao diện thì phải ghi “Mockup”, nhưng ưu tiên không thêm giao diện vì chưa có ảnh sản phẩm chạy thật.

Log có thể trình bày lại thành khung văn bản dễ đọc với nhãn “Trích log trong báo cáo”; không tạo ảnh terminal giả.

Nguồn trong chân slide ghi mục hoặc phụ lục tương ứng của tài liệu. Không tự bịa tài liệu tham khảo, DOI hoặc trích dẫn học thuật.

## 5. Kiểm tra trước khi xuất

Bảo đảm đủ 14 slide; hai hình không bị méo, cắt chữ hoặc mất nhánh. Các số 15.714 sản phẩm, 23 cột, 15.714 vector, 384 chiều, batch 256, 14 cặp ví dụ và K=5 phải nhất quán.

Kiểm tra mọi chữ “hoàn thành”, “thành công”, “100%” đều chỉ rõ đang nói đến công việc nào. Không để người xem hiểu rằng chatbot đã chạy hoàn chỉnh trong khi báo cáo vẫn ghi chưa kiểm chứng end-to-end.

Xuất slide có nội dung chỉnh sửa được và ghi chú thuyết trình đầy đủ.
