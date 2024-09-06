# ライブラリのインポート
import sys
import os

# 現在のディレクトリをシステムパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 自作関数はそれぞれfunctionフォルダに格納しています
from function import camera  # カメラを使う
from function import fish_contour  # さかなの輪郭を使った処理(抽出・部位推定)
from function import createobj as obj  # 3DOBJを作る
from function import bake_texture as texture # UV関係を作る
from function import settimer  # 時間測定
from function import painttool as paint  # ペイントツール
from function import module # 共通するモジュール

def main():
    """すべての始まり"""
    print("============Start Program============\n")
    module.initmain()

    time_d = []

    #? ------------------------------------------------------
    #? カメラ起動->画像入手
    #? ------------------------------------------------------
    # print("------------Camera in Use------------\n")
    # cameTimer = settimer.timer("-->camera-Timer")

    # camera.camera_start()

    # cameTimer.stop()
    # print("----------Camera Not in Use----------\n")
    
    #? ------------------------------------------------------
    #? ペイントツール->画像入手
    #? ------------------------------------------------------
    print("----------paint tool in Use----------\n")
    painttimer = settimer.timer("-->paint-Timer")

    paint.main()

    painttimer.stop()
    print("----------paint Not in Use----------\n")

    for i in range(1, 2):
        alltimer = settimer.timer("-->ALL-TIMER")
        file = "fish_data(" + str(i) + ").png"
        # file = "template(" + str(i) + ").png"

        #? ------------------------------------------------------
        #? 画像から魚抽出->輪郭抽出
        #? ------------------------------------------------------
        print("----------Preprocessing IMG---------\n")
        preproTimer = settimer.timer("-->Preprocessing-Timer")

        filePath = "./inputimg/" + file
        img = fish_contour.findFishParts(filePath, file)
        
        import cv2
        cv2.imwrite("./static/images/new.png", img)

        preproTimer.stop()
        print("----------End Preprocessing---------\n")

        #? ------------------------------------------------------
        #? 3DOBJに変換
        #? ------------------------------------------------------
        print("-----------Creating 3DOBJ----------\n")
        createTimer = settimer.timer("-->create-Timer")

        # 拡張子を除くファイル名を作る
        file = os.path.splitext(file)[0]
        filePath = "./output/mask/" + file + "_mask3.png"
        maskPath = "./output/mask/" + file + "_mask3.png"
        obj.creating3D(filePath, maskPath, file, 100, False)
        
        # UVを貼る
        meshpath = "./output/mesh/me_" + file + ".obj"
        onlyFishImg = "./output/onlyfish/only" + file + ".png"
        texture.makeUVs(meshpath,onlyFishImg,meshpath,file)
        
        createTimer.stop()
        print("----------- End Creating ----------\n")
        
        # 時間を終了してlistに追加する
        time_d.append(alltimer.stop())

    average_time = sum(time_d) / len(time_d)
    print(f"AVERAGE-->[{average_time:.1f}秒]")

    print("\n==============End Program=============\n")


if __name__ == "__main__":
    main()
