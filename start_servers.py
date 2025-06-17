import subprocess
import sys
import os
import webbrowser
from threading import Thread
import time

def run_flask():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, 'app.py'])
    # subprocess.run([sys.executable, 'cors_working.py'])

def run_frontend():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, '-m', 'http.server', '8000'])

if __name__ == '__main__':
    # Start Flask server in a separate thread
    flask_thread = Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    # Wait a moment for Flask to start
    time.sleep(2)

    # Start frontend server in a separate thread
    frontend_thread = Thread(target=run_frontend)
    frontend_thread.daemon = True
    frontend_thread.start()

    # Open the browser
    webbrowser.open('http://localhost:8000/login.html')

    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        sys.exit(0) 

# 医生，我感冒了，头疼。