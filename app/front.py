import os
import subprocess
import webbrowser
from threading import Timer
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    # HTMLテンプレート。ドラッグアンドドロップとボタン実行の両方を含む
    return render_template_string('''
<html>
<head>
    <title>Image Upload and Script Execution</title>
    <script>
        // ドラッグアンドドロップ関連のスクリプト
        function handleDrop(e) {
            e.preventDefault();
            e.stopPropagation();

            let dt = e.dataTransfer;
            let files = dt.files;

            uploadFile(files[0]);
        }

        function uploadFile(file) {
            let url = 'upload';
            let formData = new FormData();

            formData.append('file', file);

            fetch(url, {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                console.log(data);
                alert("Image uploaded successfully!");
            })
            .catch(error => {
                console.error(error);
            });
        }

        function setup() {
            let dropArea = document.getElementById('drop-area');

            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, preventDefaults, false);
            });

            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }

            dropArea.addEventListener('drop', handleDrop, false);
        }

        // スクリプト実行関連のスクリプト
        function runScript() {
            fetch('/run-script', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    console.log(data);
                    alert("Script executed successfully!");
                })
                .catch(error => {
                    console.error('Error:', error);
                });
        }

        window.onload = setup;
    </script>
</head>
<body>
    <div id="drop-area" style="border: 2px dashed #ccc; width: 300px; height: 200px; text-align: center; line-height: 200px;">
        Drag & Drop Images Here
    </div>
    <button onclick="runScript()">Run main.py</button>
</body>
</html>
    ''')

def open_browser():
      webbrowser.open_new('http://127.0.0.1:5000/')

@app.route('/upload', methods=['POST'])
def file_upload():
    file = request.files['file']
    filename = file.filename

    # ファイルを保存するディレクトリを指定
    upload_folder = '/Users/mypc/Desktop'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    file.save(os.path.join(upload_folder, filename))

    return jsonify({"message": "File uploaded successfully"})

@app.route('/run-script', methods=['POST'])
def run_script():
    # main.pyを実行
    subprocess.run(["/usr/bin/python3", "main.py"])
    return jsonify({"message": "Script executed successfully"})

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(debug=True)