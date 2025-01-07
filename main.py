# ライブラリのインポート
import sys
import os
import colorama
from colorama import Fore, Back, Style
import cv2
import random
import zipfile

# 現在のディレクトリをシステムパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 自作関数はそれぞれfunctionフォルダに格納しています
from function import camera  # カメラを使う
from function import extraction_fish  # さかなの輪郭を使った処理(抽出・部位推定)
from function import createobj as obj  # 3DOBJを作る
from function import make_texture as texture  # UV関係を作る
from function import settimer  # 時間測定
from function import painttool as paint  # ペイントツール
from function import module  # 共通するモジュール
from function import segmentation as sg # セグメンテーションを行う
from function import born_3d
g_upnum = 1

def regitFish(g_fishNum=0, g_templateNum=1):
    """魚の登録"""
    module.printTerminal("魚を生成")

    file, g_templateFlag = module.regitFishPath(g_fishNum, g_templateNum)
    print(file)

    # ? ------------------------------------------------------
    # ? 画像から魚抽出->輪郭抽出
    # ? ------------------------------------------------------
    module.printTerminal("画像から魚を抽出", 2)
    preproTimer = settimer.timer("-->Preprocessing-Timer")

    filePath = "./inputimg/" + file
    img = extraction_fish.findFishParts(filePath, file)

    cv2.imwrite("./static/images/new.png", img)

    preproTimer.stop()

    # ? ------------------------------------------------------
    # ? 3DOBJに変換
    # ? ------------------------------------------------------
    module.printTerminal("3DOBJを作成", 2)

    createTimer = settimer.timer("-->create-Timer")

    # 拡張子を除くファイル名を作る
    file = os.path.splitext(file)[0]
    filePath = "./output/mask/" + file + "_mask3.png"
    maskPath = "./output/mask/" + file + "_mask3.png"
    obj.creating3D(filePath, maskPath, file, 100, False)

    # UVを貼る
    meshpath = "./output/mesh/me_" + file + ".obj"
    onlyFishImg = "./output/onlyfish/only" + file + ".png"
    texture.makeUVs(meshpath, onlyFishImg, meshpath, file)

    createTimer.stop()

    # # ? ------------------------------------------------------
    # # ? ボーンを生成
    # # ? ------------------------------------------------------
    # module.printTerminal("ボーンを生成", 2)

    # bornTimer = settimer.timer("-->born-Timer")
    
    # # セグメンテーションから関節部分を推定
    # mask = cv2.imread(maskPath,cv2.IMREAD_GRAYSCALE)
    # joint_num = sg.SSFromIMG(mask,file)

    # # 関節の数を基にボーンリストを作成
    # born_list = born_3d.MakeBornList(file,joint_num)
    # print(born_list)
    
    # # ボーン情報をtxtで出力
    # born_3d.create_bone_structure(born_list,file,"output/born/")

    # bornTimer.stop()
    
    #? ------------------------------------------------------
    #? 管理番号の設定
    #? ------------------------------------------------------
    # flagがTrueならテンプレートを使用
    if g_templateFlag == True:
        # さかなをランダムな数で拡大・縮小する
        obj.scaling_obj(meshpath,module.rand_rete(2.0,0.3))

    #? ------------------------------------------------------
    #? さかなのzip化
    #? ------------------------------------------------------
    module.printTerminal("魚をzip化(送信用フォルダ化)",2)
    zipFish(meshpath,"./output/onlyfish/up2down_" + file + ".png")
    
def zipFish(fishPath, pngPath):
    """魚をZIP化します"""
    global g_upnum
    
    # パスを取得
    zipPath = f"./upload/{g_upnum}.zip"
    g_upnum += 1
    
    # 情報をターミナルに出力
    print(fishPath)
    print(pngPath)
    
    # 必須ファイルの存在を確認
    if not os.path.exists(fishPath) or not os.path.exists(pngPath):
        return print("ERR===")
    
    # ZIPファイルを作成
    with zipfile.ZipFile(zipPath, 'w') as zipf:
        zipf.write(fishPath, os.path.basename(fishPath))  # 魚のOBJ
        zipf.write(pngPath, os.path.basename(pngPath))    # 魚の画像

def save_result_to_file(result, file_path="./result.txt"):
    """result結果をtxtに格納

    Args:
        result : 生成時間の結果
        file_path : ファイルパス. Defaults to "./result.txt".
    """
    try:
        # ファイルに書き込み
        with open(file_path, "w") as file:
            for item in result:
                file.write(f"{item}\n")
        print(f"Results saved to {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    # ターミナルに色付き文字を出力するための設定
    colorama.init()

    # プログラムの開始を宣言
    module.printTerminal("Start Program")

    # 初期設定
    module.initmain()

    result = []
    
    # さかなをせいせい(nセット)
    for j in range(1000):
        # 魚を生成する(10種類)
        for i in range(1,11):
            timer = settimer.timer("-->Timer")
            regitFish(0,i)
            result.append(timer.stop())
    
    save_result_to_file(result)
    print(result)

if __name__ == "__main__":
    main()
