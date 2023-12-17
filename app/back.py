from flask import Flask, request, jsonify
import os

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True)