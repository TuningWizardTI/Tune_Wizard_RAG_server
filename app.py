from flask import Flask, request, jsonify
from rag_chain import get_rag_response

app = Flask(__name__)

@app.route("/rag", methods=["POST"]) 
def rag_handler():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"error": "query is required"}), 400

    try:
        answer = get_rag_response(query)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500 

        


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8088, debug=True)