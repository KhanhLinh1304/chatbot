from flask import Flask, request, jsonify

# import chatbot
from chatbot.chat_api import get_chat, get_patterns_from_json  # Import the function to get response


def registryRouter(app):
    @app.route("/api/v1/chat", methods=["POST"])
    def chat():
        try:
            # Lấy tin nhắn từ request
            msg = request.json["prompt"]
            print("prompt = ", msg)
            # Gọi hàm xử lý từ chatbot để lấy phản hồi
            chat_response = get_chat(msg)

            return jsonify({"text": chat_response}), 200
        except Exception as e:
            # Nếu có lỗi xảy ra, trả về mã lỗi và thông báo
            return jsonify({"error": str(e)}), 500
        #list suggest
    @app.route("/api/v1/tags", methods=["GET"])
    def get_tags():
        # Lấy đường dẫn file từ query string, nếu không có thì dùng mặc định
        file_path = request.args.get("file_path", "chatbot/intents.json")

        # Lấy các tag từ file
        tags = get_patterns_from_json(file_path)

        if tags:
            return jsonify({"tags": tags}), 200
        else:
            return (
                jsonify(
                    {"error": "Không tìm thấy tag hợp lệ hoặc lỗi trong việc đọc file."}
                ),
                400,
            )