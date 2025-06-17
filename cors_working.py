from flask import Flask, request, jsonify
import os
import requests
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Load DeepSeek API key
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'YOUR_DEEPSEEK_API_KEY')

# Manual CORS handling - guaranteed to work
@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Max-Age'] = '86400'
    print(f"Added CORS headers to {request.method} {request.path}")
    return response

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    print(f"Received {request.method} request to /api/chat")
    print(f"Origin: {request.headers.get('Origin', 'None')}")
    
    if request.method == 'OPTIONS':
        print("Handling OPTIONS preflight request")
        return '', 200
    
    data = request.get_json()
    message = data.get('message', '')
    print(f"Message received: {message}")
    
    if not message:
        return jsonify({'error': 'No message provided'}), 400

    # Wrap the message as instructed
    prompt = (
        "Act as a medical assistant. Answer the following medical question as the patient requested, "
        "and restrict your responses to professional medical answers only (don't answer otherwise):\n"
        f"{message}"
    )

    url = 'https://api.deepseek.com/v1/chat/completions'
    headers = {
        'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': 'deepseek-chat',
        'messages': [{'role': 'user', 'content': prompt}]
    }

    try:
        print("Calling DeepSeek API...")
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        print(f"DeepSeek API Status Code: {response.status_code}")
        print(f"DeepSeek API Response Headers: {dict(response.headers)}")
        
        response.raise_for_status()
        
        response_data = response.json()
        print(f"DeepSeek API Response: {response_data}")
        
        ai_response = response_data['choices'][0]['message']['content']
        print(f"DeepSeek responded successfully")
        return jsonify({'response': ai_response})
    except requests.exceptions.HTTPError as e:
        print(f'DeepSeek API HTTP Error: {e}')
        print(f'Response content: {response.text}')
        return jsonify({'response': "Sorry, I'm having trouble connecting to the AI right now."}), 500
    except KeyError as e:
        print(f'DeepSeek API Response Parsing Error: {e}')
        print(f'Full response: {response.json()}')
        return jsonify({'response': "Sorry, I received an unexpected response format from the AI."}), 500
    except Exception as e:
        print(f'DeepSeek API General Error: {e}')
        print(f'Error type: {type(e).__name__}')
        return jsonify({'response': "Sorry, I'm having trouble connecting to the AI right now."}), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'Flask server is running with CORS!'})

if __name__ == '__main__':
    print("Starting Flask server on port 5050...")
    print("Test URL: http://localhost:5050/test")
    print(f"DeepSeek API Key loaded: {'Yes' if DEEPSEEK_API_KEY != 'YOUR_DEEPSEEK_API_KEY' else 'No - check .env file'}")
    app.run(debug=True, host='0.0.0.0', port=5050) 