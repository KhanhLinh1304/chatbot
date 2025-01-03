import random
import json
import numpy as np  # type: ignore # Import thư viện numpy
import torch  # type: ignore
from chatbot.my_model import NeuralNetwork
from chatbot.nltk_utils import bag_of_words, tokenize

# Kiểm tra xem có sử dụng GPU được hay không, nếu không thì sử dụng CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Đọc file intents.json chứa các câu trả lời mẫu
with open("chatbot/intents.json", encoding="utf-8") as f:
    intents = json.load(f)

# Tải dữ liệu đã huấn luyện trước đó
FILE = "chatbot/data.pth"
data = torch.load(
    FILE, weights_only=True
)  # Tải file lưu trữ thông tin mô hình và dữ liệu

# Trích xuất thông tin từ dữ liệu đã huấn luyện
input_size = data["input_size"]  # Số lượng đặc trưng đầu vào
hidden_size = data["hidden_size"]  # Số lượng nơ-ron trong tầng ẩn
output_size = data["output_size"]  # Số lượng đầu ra
all_intents = data["all_intents"]  # Danh sách từ khóa/vocabulary
tags = data["tags"]  # Các nhãn/tag trong bộ dữ liệu
model_state = data["model_state"]  # Trạng thái mô hình đã huấn luyện

# Khởi tạo mô hình và tải trạng thái đã huấn luyện
model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)  # Tải trọng số vào mô hình
model.eval()  # Đặt mô hình ở chế độ "dự đoán" thay vì "huấn luyện"

# Tên bot và khởi động chương trình
def get_chat(sentence):
    sentence = tokenize(sentence)  # Tách câu thành từ/cụm từ (tokenize)
    X = bag_of_words(sentence, all_intents)  # Chuyển câu thành vector bag-of-words
    X = np.array(X)
    X = X.reshape(1, X.shape[0])  # Định dạng lại thành ma trận 2D (batch size = 1)
    X = torch.from_numpy(X).to(
        device, dtype=torch.float32
    )  # Chuyển sang tensor và đưa lên thiết bị phù hợp

    # Dự đoán kết quả bằng mô hình
    output = model(X)  # Mô hình trả về các điểm số cho từng nhãn
    _, predicted = torch.max(output, dim=1)  # Lấy nhãn có điểm số cao nhất
    tag = tags[predicted.item()]  # Lấy nhãn tương ứng từ danh sách tags

    # Tính xác suất dự đoán
    probs = torch.softmax(output, dim=1)  # Chuyển đổi điểm số thành xác suất
    prob = probs[0][predicted.item()]  # Xác suất của nhãn được dự đoán

    # Kiểm tra ngưỡng xác suất
    if prob.item() > 0.6:  # Nếu xác suất đủ cao (trên 60%)
        for intent in intents["intents"]:  # Duyệt qua danh sách intents trong file JSON
            if tag == intent["tag"]:  # So khớp nhãn dự đoán với tag trong intents
                responses = intent["responses"]
                return random.choice(responses)  # Lấy danh sách câu trả lời

    else:
        return "Tôi không hiểu câu hỏi của bạn, mời bạn đặt lại câu hỏi."  # Nếu không chắc chắn, trả lời mặc định
def get_patterns_from_json(file_path):
    try:
        with open(file_path, encoding="utf-8") as f:
            intents = json.load(f)  # Đọc dữ liệu JSON
        # Danh sách các tag cần lọc
        allowed_tags = ["Thanh toán", "Khoá học", "Bài học", "Tham gia thi", "Thể loại"]
        # Lọc các tag theo danh sách cho phép
        tags = [
            intent["tag"]
            for intent in intents["intents"]
            if intent["tag"] in allowed_tags
        ]
        return tags
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []