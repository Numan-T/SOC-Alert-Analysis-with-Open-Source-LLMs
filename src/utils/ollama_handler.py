import requests
import time
import subprocess
import os


def restart_ollama():
    """
    Restarts the ollama server and waits until it is ready for up to 30 seconds.
    """

    print("Restarting ollama server.")
    # Stop ollama
    try:
        os.system("taskkill /f /im ollama_app.exe >nul 2>&1")
        time.sleep(1)
    except:
        pass

    # Start ollama
    subprocess.Popen(["ollama", "serve"],
                     creationflags=subprocess.CREATE_NEW_CONSOLE)

    # Wait until ready
    start_time = time.time()
    max_wait_s = 30
    while True:
        try:
            response = requests.get("http://localhost:11434/", timeout=5)
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        if time.time() - start_time > max_wait_s:
            print("Ollama didn't restart after {max_wait_s} seconds.")
            return False
        time.sleep(0.5)
