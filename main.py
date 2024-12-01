# ライブラリのインポート
import sys
import os
import colorama
from colorama import Fore, Back, Style

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
    import cv2

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

    # ? ------------------------------------------------------
    # ? ボーンを生成
    # ? ------------------------------------------------------
    module.printTerminal("ボーンを生成", 2)

    bornTimer = settimer.timer("-->born-Timer")
    
    # セグメンテーションから関節部分を推定
    mask = cv2.imread(maskPath,cv2.IMREAD_GRAYSCALE)
    joint_num = sg.SSFromIMG(mask,file)

    # 関節の数を基にボーンリストを作成
    born_list = born_3d.MakeBornList(file,joint_num)
    
    

    bornTimer.stop()

def main():
    # ターミナルに色付き文字を出力するための設定
    colorama.init()

    # プログラムの開始を宣言
    module.printTerminal("Start Program")

    # 初期設定
    module.initmain()

    # 魚を生成する
    regitFish()


if __name__ == "__main__":
    main()
