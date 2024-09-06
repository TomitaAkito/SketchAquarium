# ライブラリのインポート
import cv2
from skimage import measure
import numpy as np


#? ---------------------------------------------------------------------------------------------------
#? 成田智史，井尻敬，“最小切断面を利用した2値画像の意味的領域分割”，画像電子学会誌，第47巻,
#? 第4号通巻246号，pp.433-439，(2018)．
#? 上記のアルゴリズムを参考に再現したもの
#? ---------------------------------------------------------------------------------------------------

#* ======================================================================================== 
#* グローバル変数
#* ======================================================================================== 
g_splitNum = []  # 収縮回数を記載


def split(img,filename):
    """結合領域を大まかに分割する

    Args:
        img (画像): マスク画像
        filename: ファイルの名前
        
    Returns:
        分割した枚数
    """
    global g_splitNum
    
    # 収縮処理用カーネル
    # 半径2画素の円形カーネルを作成
    radius = 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
    
    # 縮小用画像をコピーする
    img_erosion2 = img.copy()
    i = 0
    spNum = 0
    
    while True:
        # 画像を収縮する
        img_erosion2 = cv2.erode(img_erosion2, kernel, iterations=2)
        spNum += 1
        
        # 結合部分をラベリング
        labeled_regions, num_labels = measure.label(img_erosion2, return_num=True, connectivity=2)
        
        if num_labels >= 2:
            # 各領域のプロパティを計算
            regions = measure.regionprops(labeled_regions)
            
            # 面積の小さい領域を抽出
            min_region = min(regions, key=lambda r: r.area)
            
            # 小さい領域のみ抽出（ラベリングされた領域をマスクとして使用）
            min_mask = np.zeros_like(img)
            min_mask[labeled_regions == min_region.label] = 255
            
            # 小さい領域を保存
            file_small = './output/ss/'+ 'sp_'+  filename +'_small_' + str(i) + '.png'
            cv2.imwrite(file_small, min_mask)
            
            # 元の画像から小さい領域を削除
            img_erosion2[labeled_regions == min_region.label] = 0
            
            # マスクを保存
            file_mask = './output/ss/'+ 'sp_'+  filename +'_'+ str(i) + '.png'
            cv2.imwrite(file_mask, img_erosion2)
            
            # 回数の設定
            i += 1
            g_splitNum.append(spNum)
        
        elif num_labels == 0:
            break
        
    return i


def regrow(img,num,filename):
    """Split処理で分断された領域を膨張する

    Args:
        img (画像): マスク画像
        num (int): 分断枚数
        filename: ファイルの名前
    """
    global g_splitNum
    
    # 膨張用カーネル
    # 半径2画素の円形カーネルを作成
    radius = 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius*2+1, radius*2+1))
    
    for i in range(num):
        # 大きい画素を持っている画像を読み込む
        file = './output/ss/'+ 'sp_'+ filename +'_'+ str(i) + '.png'
        largeImg = cv2.imread(file)
        
        # 小さい画素を持っているほうを読み込む
        file = './output/ss/'+ 'sp_'+  filename +'_small_' + str(i) + '.png'
        smallImg = cv2.imread(file)
        
        # 二つのマスク画像を論理和 (OR) 演算で合成する
        combined_mask = cv2.bitwise_or(largeImg, smallImg)
        
        # 何回収縮処理をしたかリストからとってくる
        splitNum = g_splitNum[i]
        
        dilateLargeImg = largeImg.copy()
        dilateSmallImg = smallImg.copy()
        
        for j in range(splitNum):
            dilateLargeImg = cv2.dilate(dilateLargeImg, kernel, iterations=2)
            dilateSmallImg = cv2.dilate(dilateSmallImg, kernel, iterations=2)
            
        
        file_mask = './output/ss/'+ 're_' + filename +'_'+ str(i) + '.png'
        cv2.imwrite(file_mask, dilateLargeImg)
        file_mask = './output/ss/'+ 're_' + filename +'_small_'+ str(i) + '.png'
        cv2.imwrite(file_mask, dilateSmallImg)
        
        # 共通部分の計算（論理積 AND）
        common_mask = cv2.bitwise_and(dilateLargeImg, dilateSmallImg)
        file_mask = './output/ss/'+ 're_' + filename +'_common_'+ str(i) + '.png'
        cv2.imwrite(file_mask, common_mask)
    

def SSFromIMG(maskimg,filename):
    """マスク画像から意味的分割(セマンティックセグメンテーション)を行う

    Args:
        maskimg (画像): マスク画像
    """
    # 結合領域を大まかに分割するsplit処理
    maskNum = split(maskimg,filename)
    
    # もし結合結果が0であれば終了する
    if maskNum == 0:
        return
    
    # Split処理で分断された領域を膨張するRegrow処理
    regrow(maskimg,maskNum,filename)
    
    # Regrow処理の結果を用いて連結領域における面積最少の切断面(くびれ)を検索するMinCut処理
    

if __name__ == "__main__":
    imgPath = "./output/mask/fish_data(1)_mask3.png"
    img=cv2.imread(imgPath,cv2.IMREAD_GRAYSCALE)
    filename = "fish_data(1)"
    SSFromIMG(img,filename)
