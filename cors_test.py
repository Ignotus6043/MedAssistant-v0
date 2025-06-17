from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:8000"])

@app.route('/test', methods=['POST'])
def test():
    return jsonify({'message': 'CORS test successful'})

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Note: port 5001