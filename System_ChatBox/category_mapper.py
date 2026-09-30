"""
category_mapper.py
Mô-đun RUNTIME — dùng chỉ mục nhỏ đã build sẵn bởi build_category_index.py
(chạy 1 lần offline) để ánh xạ category tiếng Việt do NeedExtractor trích
xuất (vd "thảm trải sàn", "nồi nấu", "ghế sofa") sang các leaf_category
TIẾNG ANH THẬT đang tồn tại trong bảng `products` (vd "Area Rugs",
"Stockpots"/"Saucepans"/"Dutch Ovens", "Sofas & Couches"...).

Vì sao trả về NHIỀU leaf_category thay vì chỉ 1:
  Một từ tiếng Việt như "nồi" rộng hơn 1 leaf_category tiếng Anh cụ thể —
  nó có thể tương ứng với nhiều leaf_category khác nhau (Stockpots,
  Saucepans, Dutch Ovens, Pressure Cookers...). Trả về top-N category gần
  nghĩa nhất rồi lọc SQL bằng `leaf_category IN (...)` cho kết quả ĐÚNG hơn
  là ép về đúng 1 category (dễ bỏ sót) hoặc LIKE trực tiếp câu tiếng Việt
  (không bao giờ khớp — đây chính là lỗi gốc đã phát hiện).

KHÔNG gọi thêm Qwen — chỉ dùng lại SentenceTransformer đã nạp sẵn (giống
vector_search.py), nên KHÔNG phá nguyên tắc "đúng 2 lần gọi Qwen/lượt chat"
đã thống nhất trong thiết kế.

Nếu chỉ mục category chưa được build (chưa chạy build_category_index.py)
hoặc không có match nào đủ tin cậy (dưới ngưỡng), trả về [] — RecommendationService
(rule_based_filter trong main.py) khi đó sẽ KHÔNG áp lọc cứng category (đúng
tinh thần "không tự nới hay tự đặt ràng buộc khi chưa đủ căn cứ" — mục 3.5.2
khoá luận — ở đây là chiều ngược lại: không tự BỎ QUA candidate chỉ vì chưa
map được tên, để AI Matching/free_text vẫn có cơ hội xử lý).
"""

import os
import re
import unicodedata
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import chromadb
except ImportError:
    chromadb = None

CATEGORY_DB_PATH = os.getenv("CATEGORY_DB_PATH", "./category_index_db")
COLLECTION_NAME = "leaf_categories"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")

# Ngưỡng cosine similarity tối thiểu để coi 1 leaf_category là match đủ tin cậy.
# 0.30 là điểm khởi đầu THẬN TRỌNG (chưa hiệu chỉnh bằng thực nghiệm) — nhóm
# NÊN tinh chỉnh giá trị này sau khi có vài chục câu hỏi thật để so sánh, đúng
# tinh thần "cấu hình... chốt trên phần phát triển trước đánh giá chính thức"
# (mục 3.5.3 khoá luận). Ghi lại quyết định cuối khi chốt.
DEFAULT_MIN_SCORE = 0.30
DEFAULT_TOP_N = 5

_model = None
_collection = None
_warned_unavailable = False

# Cache trong phiên chạy của tiến trình — nhiều user có thể hỏi cùng 1 category
# phổ biến ("nồi", "thảm"...), không cần encode lại mỗi lần.
_cache: dict = {}

# --- MỚI: bảng gợi ý từ khóa tiếng Việt cho các danh từ NGẮN/MƠ HỒ mà vector
# similarity của model đa ngôn ngữ đang dùng (paraphrase-multilingual-MiniLM-
# L12-v2) xử lý KÉM — đã phát hiện thật qua test: "nồi"/"nồi nấu"/"nồi nấu ăn"
# đều KHÔNG đưa "Stockpots" vào top-5, thậm chí "nồi" một mình còn khớp ra
# "Vacuums" (máy hút bụi, hoàn toàn không liên quan).
#
# KHÔNG dùng cách tự động tìm chuỗi con "pot" trong leaf_category — sẽ dính
# nhầm "Potato Mashers", "Pot Racks", "Potholders", "Potpourris" (đều chứa
# "pot" nhưng chẳng liên quan nồi nấu). Danh sách dưới đây đã CHỌN TAY, xác
# minh từng tên đúng là leaf_category THẬT tồn tại trong dataset v4 (đối
# chiếu bằng script khảo sát, không đoán).
#
# Cơ chế: nếu category text (đã bỏ dấu) CHỨA 1 từ khóa dưới đây (khớp
# NGUYÊN TỪ, không phải substring thô — cùng kỹ thuật word-boundary đã
# dùng trong color_mapper.py để tránh lỗi kiểu "khong biet" khớp nhầm
# "hồng"), GỘP THẲNG danh sách category tương ứng vào kết quả — KHÔNG phụ
# thuộc ngưỡng min_score, vì đây là ánh xạ đã xác minh tay, tin cậy hơn
# điểm vector. Vector search vẫn chạy thêm để bổ sung nếu thiếu, không bị
# thay thế hoàn toàn.
#
# Đây là danh sách đã MỞ RỘNG — dịch XUÔI (Anh → Việt) qua TOÀN BỘ 1.127
# leaf_category thật của dataset v4 (không đoán ngược từ câu user, vì không
# liệt kê hết được — category thì hữu hạn, câu nói thì vô hạn). Nhóm theo
# các danh từ tiếng Việt PHỔ BIẾN khi mua sắm Home & Kitchen.
#
# QUAN TRỌNG — mức độ tin cậy KHÁC "nồi" (mục ban đầu): entry "nồi" đã được
# XÁC MINH THẬT qua test lỗi cụ thể (model trả "Vacuums" sai hoàn toàn).
# Các entry MỚI dưới đây là bản dịch BEST-EFFORT dựa trên hiểu biết ngôn
# ngữ, CHƯA được test đối chiếu với model thật cho từng entry — có thể còn
# thiếu sót (bỏ lọt category liên quan) hoặc gộp hơi rộng (thà rộng còn hơn
# thiếu, vì rule_based_filter dùng "IN (...)" — rộng thêm 1 category không
# liên quan sẽ bị lọc tiếp bởi giá/màu/AI Matching, không gây sai lệch nặng
# như thiếu hẳn category đúng). Chạy batch_test_category_mapper.py để tự
# rà thêm, báo lại nếu thấy nhóm nào ghép sai.
#
# ĐÃ CHỦ Ý BỎ QUA (không đưa vào bảng): các leaf_category dạng khuyến mãi/
# nhãn tiếp thị/rác dữ liệu không phải loại sản phẩm thật (vd "Home Outlet",
# "Kitchen Back To School | Buy 3 Get 10% Off", "New Arrivals for Your
# Home", "Frustration-Free Packaging...", "Electric", "Manual", "Portable",
# "Round", "Kits", "Chantal Trudel" — đây là tên thương hiệu, không phải
# loại sản phẩm) — đã từng phát hiện các nhóm này lúc khảo sát chất lượng
# dữ liệu trước đó.
VI_CATEGORY_HINTS: Dict[str, List[str]] = {
    # --- Dụng cụ nấu ăn ---
    "noi": ["Stockpots", "Pots & Pans", "Dutch Ovens", "Saucepans",
            "Pressure Cookers", "Slow Cookers", "Rice Cookers",
            "Multipots & Pasta Pots", "Hot Pots", "Electric Pressure Cookers",
            "Pots & Sets"],
    "chao": ["Skillets", "Grill Pans", "Crepe Pans", "Woks & Stir-Fry Pans",
             "Chef's Pans", "Sauté Pans", "Electric Skillets", "Omelet Pans",
             "Paella Pans", "Griddles", "Electric Griddles"],
    "am": ["Tea Kettles", "Electric Kettles", "Kettles & Tea Machines",
           "Teapots", "Teapots & Coffee Servers", "Teapot Warmers"],
    "dao": ["Chef's Knives", "Kitchen Knives & Accessories", "Bread Knives",
            "Boning Knives", "Carving Knives", "Cheese Knives",
            "Paring Knives", "Utility Knives", "Santoku Knives",
            "Sashimi Knives", "Deba Knives", "Fillet Knives",
            "Garnishing Knives", "Gyutou Knives", "Usuba & Nakiri Knives",
            "Cake Knives", "Butter Knives", "Vegetable Cleavers",
            "Meat Cleavers", "Dinner Knives", "Asian Knives", "Knife Sets",
            "Knife Block Sets", "Knife Blocks", "Knife Sharpeners"],
    # "keo" ĐÃ SỬA — đây là ambiguity THẬT (không phải false trigger như các
    # case đã xóa ở trên): "kéo" (kéo cắt) và "kẹo" (candy — CÓ category
    # thật "Candy" trong dataset) đều chuẩn hoá cùng về "keo", và CẢ HAI đều
    # là sản phẩm hợp lệ có thể user đang hỏi. Xử lý bằng cách GỘP CẢ 2 khả
    # năng thay vì chọn 1 (rồi để category+giá+màu+AI Matching lọc tiếp ở
    # bước sau) — an toàn hơn đoán sai hẳn 1 hướng.
    "keo": ["Shears", "Candy"],
    "thot": ["Cutting Boards"],
    "muong": ["Spoons", "Serving Spoons", "Cooking Spoons", "Dessert Spoons",
              "Soup Spoons", "Teaspoons", "Iced Tea Spoons",
              "Coffee & Espresso Spoons", "Specialty Spoons", "Caviar Spoons",
              "Chinese Spoons", "Olive Spoons"],
    "thia": ["Spoons", "Serving Spoons", "Cooking Spoons", "Dessert Spoons",
             "Soup Spoons", "Teaspoons"],
    "dua": ["Chopsticks"],
    # ĐÃ XÓA key "va" (vá/ladle) — collision NGUY HIỂM với "và" (liên từ,
    # 1 trong những từ phổ biến nhất tiếng Việt). Giá trị của "vá" (dụng cụ
    # ít được hỏi riêng lẻ) không đáng đánh đổi lấy rủi ro này.
    # ĐÃ XÓA key "bat" (bát/bowl) — collision NGUY HIỂM NHẤT trong toàn bộ
    # dict: "bắt" trùng chuẩn hoá với "bát", mà "bắt buộc" CHÍNH LÀ cụm từ
    # hệ thống này dùng làm tín hiệu hard_constraints (xem need_extractor.py)
    # — nếu user nói "bắt buộc phải màu đỏ", từ "bắt" có thể vô tình khớp
    # nhầm ra category "Bowl Sets" dù chẳng liên quan gì. "bắt đầu" (bắt đầu
    # tìm) cũng phổ biến, càng làm tăng rủi ro. Bát vẫn có thể được vector
    # search xử lý (kém tin cậy hơn nhưng an toàn hơn hẳn).
    "chen": ["Bowl Sets", "Cereal Bowls", "Rice Bowls", "Teacups",
             "Cup & Saucer Sets"],
    "dia": ["Plates", "Dinner Plates", "Dessert Plates", "Salad Plates",
            "Appetizer Plates", "Bread & Butter Plates", "Charger Plates",
            "Sushi Plates", "Commemorative & Decorative Plates", "Platters"],
    "ly": ["Wine Glasses", "Beer Glasses", "Champagne Glasses",
           "Cordial Glasses", "Shot Glasses", "Martini Glasses",
           "Margarita Glasses", "Old Fashioned Glasses", "Highball Glasses",
           "Tumblers", "Tumblers & Water Glasses", "Iced Tea Glasses",
           "Snifters", "Goblets & Chalices", "Espresso Cups",
           "Mint Julep Cups", "Irish Coffee Glasses", "Drinkware",
           "Glassware", "Glassware & Drinkware"],
    "coc": ["Cups", "Cups & Mugs", "Coffee Cups & Mugs", "Coffee Mugs",
            "Mug Sets", "Teacups", "Beer Mugs & Steins", "Moscow Mule Mugs",
            "Cups, Mugs, & Saucers"],
    "binh giu nhiet": ["Thermoses", "Thermal Carafes", "Insulated Food Jars"],
    "hop dung thuc pham": ["Food Containers", "Food Storage",
                            "Travel & To-Go Food Containers",
                            "Food Jars & Canisters", "Canisters",
                            "Cereal Containers", "Lunch Boxes", "Bento Boxes",
                            "Container Sets", "Containers"],
    "ro": ["Colanders", "Baskets, Bins & Containers"],
    "khay nuong": ["Baking & Cookie Sheets", "Bakeware", "Bakeware Sets",
                   "Muffin & Cupcake Pans", "Loaf Pans", "Pie Pans",
                   "Tart Pans", "Bundt Pans", "Springform", "Roasting Pans",
                   "Donut Pans"],
    "may xay": ["Blenders", "Countertop Blenders", "Personal Size Blenders",
                "Hand Blenders", "Food Processors", "Meat Grinders",
                "Coffee Grinders", "Grain Mills", "Herb & Spice Mills"],
    "may pha ca phe": ["Coffee Machines", "Espresso Machine & Coffeemaker Combos",
                        "Semi-Automatic Espresso Machines",
                        "Super-Automatic Espresso Machines",
                        "Pour Over Coffee Makers", "French Presses",
                        "Single-Serve Brewers", "Cold Brew Coffee Makers",
                        "Stovetop Espresso & Moka Pots"],
    "noi chien khong dau": ["Air Fryers"],
    "lo nuong": ["Ovens & Toasters", "Convection Ovens"],
    "lo vi song": ["Countertop Microwave Ovens", "Microwave Oven Replacement Parts"],
    "may nuong banh mi": ["Ovens & Toasters", "Toasters"],
    "bep": ["Gas Stoves", "Countertop Burners", "Indoor Grills & Griddles"],
    "tap de": ["Aprons"],
    "bao tay": ["Oven Mitts", "Potholders"],
    "kep gap": ["Tongs", "Ice Tongs", "Buffet Tongs"],
    "can bot": ["Rolling Pins"],
    "phoi tron": ["Whisks", "Beaters", "Hand Mixers", "Stand Mixers"],
    "bao": ["Peelers", "Graters", "Zesters", "Mandolines & Slicers"],
    # ĐÃ XÓA key "can" (cân/scale) — collision NGUY HIỂM: "cần" (động từ
    # "tôi CẦN mua...", "CẦN tìm...") là 1 trong những từ phổ biến NHẤT
    # trong câu yêu cầu mua sắm, gần như chắc chắn xuất hiện ở đầu hầu hết
    # tin nhắn — nếu giữ key này, rủi ro khớp nhầm ra "Scales" gần như mỗi
    # lượt chat là không chấp nhận được. Cân nhà bếp vẫn để vector search xử
    # lý (kém tin cậy hơn nhưng không có rủi ro false-positive diện rộng).

    # --- Đồ phòng khách / phòng ngủ / nội thất ---
    "ghe": ["Chairs", "Armchairs", "Barstools", "Folding Chairs", "Recliners",
            "Desk Chairs", "Kneeling Chairs", "Directors Chairs",
            "Video Game Chairs", "Computer Gaming Chairs",
            "Home Office Desk Chairs"],
    "ghe sofa": ["Sofas", "Sofas & Couches", "Sofa Parts", "Sofa Slipcovers",
                 "Sofa & Console Tables"],
    "ban": ["Tables", "Coffee Tables", "End Tables", "Bar Tables",
            "Nesting Tables", "Folding Tables", "Desks", "Drafting Tables"],
    "ke tivi": ["Television Stands & Entertainment Centers", "TV & Media Furniture"],
    "tu quan ao": ["Armoires & Dressers", "Bedroom Armoires", "Dressers",
                   "Garment Racks", "Closet Rods"],
    "giuong": ["Beds", "Bed Frames", "Bed Frames, Headboards & Footboards",
               "Headboards", "Daybed Sets"],
    "goi": ["Bed Pillows", "Bed Pillows & Positioners", "Throw Pillows",
            "Floor Pillows & Cushions", "Body Pillows", "Lumbar Pillows",
            "Travel Pillows", "Neck & Cervical Pillows",
            "Reading & Bed Rest Pillows", "Specialty Medical Pillows",
            "Leg Positioner Pillows", "Pillow Inserts", "Pillow Protectors",
            "Pillow Shams", "Throw Pillow Covers"],
    "chan": ["Blankets & Throws", "Bed Blankets", "Electric Blankets",
             "Weighted Blankets", "Wearable Blankets", "Throws", "Quilts",
             "Quilts & Sets", "Quilt Sets", "Comforters", "Comforters & Sets",
             "Comforter Sets", "Duvets & Down Comforters"],
    "ga trai giuong": ["Fitted Sheets", "Flat Sheets", "Sheets & Pillowcases",
                       "Sheet & Pillowcase Sets", "Bedding",
                       "Bedding Sets & Collections", "Pillowcases"],
    "rem": ["Curtains & Drapes", "Curtain Panels", "Shower Curtains",
            "Shower Curtain Sets", "Panels", "Blinds & Shades",
            "Window Treatments", "Roman Shades", "Roller Shades",
            "Horizontal Blinds", "Cellular Shades", "Panel Track Blinds",
            "Valances", "Swags"],
    "tham": ["Area Rugs", "Rugs", "Rugs, Pads & Protectors", "Area Rug Sets",
             "Bath Rugs", "Kitchen Rugs", "Runners", "Aisle Runners", "Rug Pads"],
    "den": ["Lamps & Lighting", "Candle Lamps", "Oil Lamps",
            "Indoor String Lights", "Rope Lights", "Seasonal Lighting"],
    "quat": ["Household Fans", "Floor Fans", "Table Fans", "Tower Fans",
             "Personal Fans", "Pedestal Fans", "Wall-Mounted Fans", "Window Fans"],
    "guong": ["Mirrors", "Wall-Mounted Mirrors", "Floor & Full Length Mirrors",
              "Mirror Sets", "Wall-Mounted Vanity Mirrors"],
    "ke sach": ["Bookcases", "Bookcases, Cabinets & Shelves",
                "Floating Shelves", "Corner Shelves", "Standing Shelf Units",
                "Hanging Shelves"],
    "ban ui": ["Irons", "Irons & Steamers", "Ironing Boards", "Ironing Board Covers"],
    "ban la": ["Irons", "Irons & Steamers", "Ironing Boards", "Ironing Board Covers"],

    # --- Vệ sinh / dọn dẹp ---
    "may hut bui": ["Vacuums", "Vacuums & Floor Care", "Handheld Vacuums",
                    "Upright Vacuums", "Canister Vacuums", "Robotic Vacuums",
                    "Stick Vacuums & Electric Brooms",
                    "Vacuum Parts & Accessories", "Central Vacuum Systems"],
    "thung rac": ["Wastebaskets", "Kitchen Trash Cans", "Outdoor Trash Cans",
                  "Recycling Bins", "In-Home Recycling Bins",
                  "In-Home Composting Bins", "Trash, Recycling & Compost"],
    "moc treo": ["Clothes Hangers", "Coat Hangers", "Standard Hangers",
                 "Children's Clothes Hangers", "Pants Hangers", "Suit Hangers",
                 "Key Hooks", "Coat Hooks", "Utility Hooks"],
    "khan tam": ["Bath Towels", "Hand Towels", "Beach Towels",
                 "Fingertip Towels", "Kids' Bath Towels", "Washcloths",
                 "Towel Sets", "Towel Racks", "Towel Warmers"],
    "khan": ["Bath Towels", "Hand Towels", "Dish Cloths & Dish Towels",
             "Cloth Napkins", "Washcloths"],
    "ban chai": ["Cleaning Brushes", "Brushes", "Toilet Brushes & Holders"],
    "may loc khong khi": ["Air Purifiers", "HEPA Air Purifiers",
                           "Charcoal Air Purifiers", "Travel-Size Air Purifiers"],
    "may loc nuoc": ["Pitcher Water Filters", "Water Coolers",
                     "Water Coolers & Filters", "Handheld Filters"],
    "may hut am": ["Dehumidifiers"],
    "may tao do am": ["Humidifiers"],
    "may suoi": ["Electric Space Heaters", "Propane Space Heaters"],
    "lo suoi": ["Fireplaces", "Electric Fireplaces", "Gas Fireplaces",
                "Gel & Ethanol Fireplaces", "Fireplace Screens",
                "Wood Burning Stoves", "Pellet Stoves"],
    "may dieu hoa": ["Air Conditioners", "Air Conditioner Parts & Accessories"],
    "gia phoi do": ["Drying Racks"],

    # --- Trang trí ---
    "binh hoa": ["Vases", "Vase Fillers"],
    "nen": ["Candles", "Candleholders", "Candlestick Holders", "Tea Lights",
            "Tea Light Holders", "Votive Candles", "Taper Candles",
            "Jar Candles", "Floating Candles", "Devotional Candles",
            "Birthday Candles", "Candle Sets", "Candelabras"],
    "tranh": ["Wall Art", "Paintings", "Posters & Prints", "Picture Frames",
              "Wall & Tabletop Frames", "Poster Frames"],
    "dong ho": ["Wall Clocks", "Clocks", "Mantel Clocks", "Desk & Shelf Clocks",
                "Alarm Clocks", "Cuckoo Clocks", "Sundial Clocks", "Specialty Clocks"],
}

_HINT_SORTED_KEYS = sorted(VI_CATEGORY_HINTS.keys(), key=len, reverse=True)


def _normalize_vi(text: str) -> str:
    """Bỏ dấu tiếng Việt + về chữ thường — dùng để so khớp VI_CATEGORY_HINTS
    không phân biệt hoa/thường, không phân biệt cách gõ dấu. QUAN TRỌNG: phải
    tự chuyển "đ"->"d" TRƯỚC khi strip combining mark, vì "đ" là 1 ký tự
    Unicode riêng biệt (U+0111), KHÔNG tự decompose qua NFKD như các nguyên
    âm có dấu thanh (à, ả, ã...) — nếu bỏ bước này, "đèn" sẽ chuẩn hoá thành
    "đen" (giữ nguyên đ) chứ không phải "den", khiến mọi key liên quan tới
    chữ có "đ" không bao giờ khớp được (lỗi THẬT đã phát hiện qua test)."""
    text = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text.strip().lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _lookup_hints(vi_category_text: str) -> List[str]:
    """Trả về danh sách leaf_category từ VI_CATEGORY_HINTS — GỘP kết quả của
    TẤT CẢ key khớp được (không dừng ở key đầu tiên). Bắt buộc phải gộp thay
    vì chỉ lấy 1 key: câu ghép nhiều từ như "đèn bàn" (đèn bàn = table lamp)
    chứa CẢ "đèn" lẫn "bàn" là 2 key khác nhau (đều 3 ký tự sau chuẩn hoá,
    thứ tự ai trước ai sau khi bằng độ dài không đáng tin) — nếu chỉ lấy key
    đầu tiên, có thể chọn nhầm "bàn"/Tables thay vì "đèn"/Lamps (lỗi THẬT đã
    phát hiện qua test). Gộp cả 2 vẫn còn dư 1 ít category không liên quan
    (Tables), nhưng KHÔNG BỎ SÓT category đúng — ưu tiên không bỏ sót hơn
    tuyệt đối chính xác, vì các bước lọc sau (giá/màu/AI Matching) vẫn còn
    cơ hội loại bớt category thừa.

    Khớp NGUYÊN TỪ bằng \\b, không phải substring thô — tránh lỗi tương tự
    "khong biet" khớp nhầm "hồng" đã gặp ở color_mapper. Trả về [] nếu không
    khớp key nào."""
    normalized = _normalize_vi(vi_category_text)
    matched: List[str] = []
    for key in _HINT_SORTED_KEYS:
        if re.search(rf"\b{re.escape(key)}\b", normalized):
            for cat in VI_CATEGORY_HINTS[key]:
                if cat not in matched:
                    matched.append(cat)
    return matched


def _lazy_init() -> bool:
    global _model, _collection, _warned_unavailable

    if _collection is not None:
        return True

    if SentenceTransformer is None or chromadb is None:
        if not _warned_unavailable:
            print("[category_mapper] Thiếu sentence-transformers/chromadb — "
                  "bỏ qua ánh xạ category, rule_based_filter sẽ không lọc cứng category.")
            _warned_unavailable = True
        return False

    try:
        client = chromadb.PersistentClient(path=CATEGORY_DB_PATH)
        _collection = client.get_collection(name=COLLECTION_NAME)
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return True
    except Exception as e:
        if not _warned_unavailable:
            print(f"[category_mapper] Chưa có chỉ mục category hợp lệ ({e}) — "
                  f"hãy chạy build_category_index.py trước. Tạm thời rule_based_filter "
                  f"sẽ không lọc cứng category.")
            _warned_unavailable = True
        return False


def resolve_leaf_categories(
    vi_category_text: str,
    top_n: int = DEFAULT_TOP_N,
    min_score: float = DEFAULT_MIN_SCORE,
) -> List[str]:
    """Trả về danh sách leaf_category TIẾNG ANH THẬT (đã tồn tại trong DB)
    gần nghĩa nhất với `vi_category_text` (tiếng Việt, do NeedExtractor trích
    xuất). Trả về [] nếu chưa sẵn sàng hoặc không có match nào đủ tin cậy —
    KHÔNG bao giờ bịa ra tên category không tồn tại.

    THỨ TỰ ưu tiên (MỚI):
      1. VI_CATEGORY_HINTS — bảng đã xác minh tay cho các từ mơ hồ mà vector
         search xử lý kém (xem giải thích ở khai báo VI_CATEGORY_HINTS phía
         trên). Nếu khớp, các category này LUÔN được đưa vào kết quả, KHÔNG
         phụ thuộc min_score.
      2. Vector search (như cũ) — chạy THÊM để bổ sung nếu còn thiếu, không
         thay thế hints. Nếu index chưa sẵn sàng nhưng đã có hint khớp, vẫn
         trả về đúng hint (không cần model)."""
    if not vi_category_text or not vi_category_text.strip():
        return []

    key = (vi_category_text.strip().lower(), top_n, min_score)
    if key in _cache:
        return _cache[key]

    matched: List[str] = list(_lookup_hints(vi_category_text))  # có thể rỗng

    if _lazy_init():
        try:
            query_vec = _model.encode(vi_category_text, normalize_embeddings=True)
            result = _collection.query(
                query_embeddings=[query_vec.tolist()],
                n_results=top_n,
            )
            ids = result.get("ids", [[]])[0]
            # Chroma mặc định trả "distances" (khoảng cách), KHÔNG phải
            # similarity. Với vector đã chuẩn hoá đơn vị và index cosine,
            # cosine_similarity = 1 - cosine_distance.
            distances = result.get("distances", [[]])[0]
            for cat_name, dist in zip(ids, distances):
                similarity = 1.0 - dist
                if similarity >= min_score and cat_name not in matched:
                    matched.append(cat_name)
        except Exception as e:
            print(f"[category_mapper] Lỗi truy vấn chỉ mục category: {e}")
            # Không return [] ở đây nữa — vẫn giữ kết quả từ hints (nếu có).

    _cache[key] = matched
    if not matched:
        print(f"[category_mapper] Không có leaf_category nào đủ tin cậy (>= {min_score}) "
              f"cho category='{vi_category_text}' — bỏ qua lọc cứng category ở lượt này.")
    return matched


if __name__ == "__main__":
    # Demo nhanh — cần đã chạy build_category_index.py trước.
    for demo in ["thảm trải sàn", "nồi nấu", "nồi nấu ăn", "nồi", "ghế sofa", "dao làm bếp", "gối ôm"]:
        print(demo, "->", resolve_leaf_categories(demo))