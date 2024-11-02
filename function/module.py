# ライブラリのインポート
import sys
import os
import colorama
from colorama import Fore, Back, Style

#?----------------------------------------------------
#? 自作モジュール
#?----------------------------------------------------

def namingFile(name,extension,dirPath):
    """ディレクトリ内に格納されているnameの番号を決める
    fish(1).pngがあればfish(2).pngを返す

    Args:
        name (str): ファイルの名前
        extension (str): 拡張子
        dirPath (str): directoryのパス

    Returns:
        str: ファイル名
    """
    num = 1
    while True:
        # 探索するファイル名を検索
        filePath = dirPath + "/" + name + "(" + str(num) + ")" + extension
        # directoryになければ終了
        if not os.path.isfile(filePath):
            return name + "(" + str(num) + ")" + extension
        else:
            num += 1
        
def makeFile(filePath):
    """ディレクトリがあるかを判定する．なければ作る．

    Args:
        filePath : ファイルパス
    """
    if not os.path.isdir(filePath):
        os.makedirs(filePath)
        
def initmain():
    """プログラム起動をする際の設定"""
    makeFile("./inputimg")
    makeFile("./output/mask")
    makeFile("./output/mesh")
    makeFile("./output/onlyfish")
    makeFile("./output/split")
    makeFile("./output/ss")
    makeFile("./output/paint")
    makeFile("./output/camera")
    makeFile("./upload")
    
def regitFishPath(nowNum,templateNum):
    """登録する魚を決定する

    Args:
        nowNum (int): fish_dataの番号
        templateNum (int): templateの番号

    Returns:
        パス，flag : Trueならテンプレート使用
    """
    fileBase = "./inputimg/"
    
    serchPath = fileBase + "fish_data(" + str(nowNum) + ").png"
    
    if os.path.isfile(serchPath):
        return "fish_data(" + str(nowNum) + ").png",False
    else:
        return "template(" + str(templateNum) + ").png",True
    
def getFish(fishNum,templateNum):
    """指定された番号のパスを返す

    Args:
        num (int):番号. Defaultsはg_fishNum.

    Returns:
        パス
    """
    serchPath = "./output/mesh/me_fish_data("+str(fishNum)+").obj"
    
    if os.path.isfile(serchPath):
        return serchPath
    else:
        return "./output/mesh/me_template(" + str(templateNum) + ").obj"
    
def getImgFish(fishNum,templateNum):
    """指定された番号のパスを返す

    Args:
        num (int):番号. Defaultsはg_fishNum.

    Returns:
        パス
    """
    serchPath = "./output/onlyfish/up2down_fish_data("+str(fishNum)+").png"
    
    if os.path.isfile(serchPath):
        return serchPath
    else:
        return "./output/onlyfish/up2down_template("+str(templateNum)+").png"

def printTerminal(string,color = 1):
    """何の処理をしているか表示させる

    Args:
        string (str): 表示させるもの
    """
    colorama.init()
    if color == 1:
        print(Fore.LIGHTBLUE_EX + "\n======================================================")
        print(Fore.LIGHTBLUE_EX + "  " + string)
        print(Fore.LIGHTBLUE_EX + "======================================================")

    elif color == 2:
        print(Fore.LIGHTGREEN_EX + "======================================================")
        print(Fore.LIGHTGREEN_EX + "  " + string)
        print(Fore.LIGHTGREEN_EX + "======================================================")
        
        
    print(Fore.WHITE)
    print(Style.RESET_ALL+"",end="")
    
if __name__ == "__main__":
    print(namingFile("fish_data",".png","./inputimg"))