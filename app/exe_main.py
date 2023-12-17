from flask import Flask, jsonify
import subprocess

app = Flask(__name__)

@app.route('/run-script', methods=['POST'])
def run_script():
    # ここで main.py を実行
    try:
        subprocess.run(["python", "main.py"], check=True)
        return jsonify({"message": "Script executed successfully"})
    except subprocess.CalledProcessError as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)