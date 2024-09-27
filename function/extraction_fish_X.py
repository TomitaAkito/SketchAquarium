import cv2
import numpy as np


def calcSlope(x1, y1, x2, y2):
    return (y1 - y2) / (x1 - x2)


def contourDetection(imgPath, filename):
    """
    画像から魚のみを抽出する

    ------------------
    【入力】
    imgPath: 画像のpath
    filename: ファイルの名前
    """
    # 画像をグレースケールで読み込む
    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)

    # 二値化処理を行う
    ret, img_2 = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    cv2.imwrite("./mask/" + filename + "_mask1.png", img_2)

    # 二値化画像の色を反転
    img_2re = cv2.bitwise_not(img_2)

    # 輪郭を抽出する
    contours, hierarchy = cv2.findContours(
        img_2re, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    # マスク画像を作成し、輪郭部分を塗りつぶす
    mask = np.zeros_like(img)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    cv2.imwrite("./mask/" + filename + "_mask2.png", mask)

    # カラー画像を読み込む
    img_color = cv2.imread(imgPath)

    # 最大の輪郭を見つける
    max_contour = max(contours, key=cv2.contourArea)

    # 輪郭の内部を白で塗りつぶす
    cv2.drawContours(mask, [max_contour], -1, (255), thickness=cv2.FILLED)

    # マスクをカラー画像に変換
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.imwrite("./mask/" + filename + "_mask3.png", mask_color)

    # マスクを使って元の画像から輪郭部分を抽出
    img_cut = cv2.bitwise_and(img_color, mask_color)

    return img_cut


def findWidestPoint(contour):
    """
    輪郭から最も幅の広い箇所のx座標を返す

    ------------------
    【入力】
    contour: 輪郭(body)
    """
    # 輪郭のx座標の最小値と最大値を取得
    x_min, x_max = contour[:, :, 0].min(), contour[:, :, 0].max()

    max_width = 0
    widest_x = 0

    # x座標ごとに輪郭の幅を計算し、最も幅が広い部分を特定する
    for x in range(x_min, x_max):
        y_values = contour[contour[:, :, 0] == x][:, 1]
        if len(y_values) > 1:
            width = y_values.max() - y_values.min()
            if width > max_width:
                max_width = width
                widest_x = x

    widest_x = x_min + (x_max - x_min) // 2
    return widest_x


def findNarrowestPointAfter(contour, start_x):
    """
    指定されたx座標以降の最も幅が狭くなる部分を特定する

    ------------------
    【入力】
    contour: 輪郭(body)
    start_x: x座標
    """

    # 探索範囲をx座標の最大値までに設定
    x_max = contour[:, :, 0].max()

    min_width = float("inf")
    split_x = start_x
    before_xy = [-999, -999]
    minUp = 0
    spUp = 0
    flag = False

    # 最も幅が広い部分から始めて、最も幅が狭くなる部分を特定する
    # なお，途中から前の幅より大きくなった場合はそこを返す
    for x in range(start_x, x_max):
        y_values = contour[contour[:, :, 0] == x][:, 1]
        if len(y_values) > 0:
            width = y_values.max() - before_xy[1]
            if width < 0:
                width *= -1
            if min_width > width:
                min_width = width
                split_x = x

            # if flag == False and minUp < y_values:
            #     minUp = y_values
            #     spUp = x

            # if flag == False and

            print(before_xy)
            print(x, y_values.max())
            print(width, min_width, end="\n-----------------\n")

            before_xy[0] = x
            before_xy[1] = y_values.max()

    return split_x


def findFishParts(imgPath, filename):
    """
    画像から魚のパーツを推定する

    ------------------
    【入力】
    imgPath: 画像のパス
    filename: 保存する名前
    """

    # 輪郭を抽出して画像を切り抜く
    img_cut = contourDetection(imgPath, filename)

    # 切り抜いた画像をグレースケールに変換
    gray = cv2.cvtColor(img_cut, cv2.COLOR_BGR2GRAY)

    # 再度二値化処理
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # 輪郭を再度検出
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 最大の輪郭を魚の輪郭と見なす
    fish_contour = max(contours, key=cv2.contourArea)

    # 輪郭の周囲の長さを計算する
    arclen = cv2.arcLength(fish_contour, True)

    # 輪郭を近似する
    epsilon = 0.005 * arclen
    a_fish_contour = cv2.approxPolyDP(fish_contour, epsilon, closed=True)

    print(a_fish_contour)

    # 最も幅が広い部分を特定
    widest_point = findWidestPoint(a_fish_contour)

    # 身体の分割線を決定
    split_line = findNarrowestPointAfter(a_fish_contour, widest_point)

    body_contour = []
    tail_contour = []

    # 輪郭の点をボディ部分としっぽ部分に分ける
    for point in fish_contour:
        if point[0][0] < split_line:
            body_contour.append(point)
        else:
            tail_contour.append(point)

    body_contour = np.array(body_contour)
    tail_contour = np.array(tail_contour)

    # 結果を保存
    cv2.imwrite("./img/" + filename + "_BSfishparts.png", img_cut)

    # ボディ部分を青色で描画
    cv2.drawContours(img_cut, [body_contour], -1, (255, 0, 0), 2)

    # しっぽ部分を赤色で描画
    cv2.drawContours(img_cut, [tail_contour], -1, (0, 0, 255), 2)

    # 分割ラインを黄色で描画
    y_min, y_max = fish_contour[:, 0, 1].min(), fish_contour[:, 0, 1].max()
    cv2.line(img_cut, (split_line, y_min), (split_line, y_max), (0, 255, 255), 2)

    # 輪郭のx座標の最小値と最大値を描画
    x_min, x_max = fish_contour[:, :, 0].min(), fish_contour[:, :, 0].max()
    cv2.line(img_cut, (x_min, y_min), (x_min, y_max), (255, 255, 255), 2)
    cv2.line(img_cut, (x_max, y_min), (x_max, y_max), (255, 255, 255), 2)

    # 探索開始位置を描画
    cv2.line(img_cut, (widest_point, y_min), (widest_point, y_max), (60, 170, 255), 2)

    # 結果を保存
    cv2.imwrite("./img/" + filename + "_fishparts.png", img_cut)

    return contours

if __name__ == "__main__":
    print('main.pyを実行してください')