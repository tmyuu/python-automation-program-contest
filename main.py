import runpy
from categoryDetection import category_detection
from extract_mercariImages import mercari_scraping

runpy.run_path("./extract_zozoImages.py")
runpy.run_path("./generate_zozoPkl.py")
runpy.run_path("./trainingModel.py")

# category_detection関数を実行し、結果を取得
result = category_detection()

# 分類されたカテゴリを文字列型で取得
response = str(result['response'])

# mercari_scraping関数に分類されたカテゴリを引数として渡す
mercari_scraping(response)

# その他のスクリプトを実行
runpy.run_path("./generate_mercariPkl.py")
runpy.run_path("./similarDetection.py")
