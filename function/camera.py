import cv2
import numpy as np
import os
import sys
# 現在のディレクトリをシステムパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import module


def removeShade(img):
    """画像から影を除去する

    Args:
        img (画像): 除去する画像

    Returns:
        im_shadow: 影除去した画像
    """
    # RGB分割
    rgb_planes = cv2.split(img)
    result_planes = []

    for plane in rgb_planes:
        # 膨張処理
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        # メディアンフィルタ
        bg_img = cv2.medianBlur(dilated_img, 21)
        # ノイズ除去と元画像の差分
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(diff_img)

    # 3つの差分画像を結合
    im_shadow = cv2.merge(result_planes)

    return im_shadow


def transform_by4(img, points):
    """
    画像を透視変換を用いて切り抜く

    Args:
        img: 画像
        points: 座標

    Returns:
        切り抜かれた画像
    """
    # ポイントをy座標でソートし、上部と下部に分ける
    points = sorted(points, key=lambda x: x[1])
    top = sorted(points[:2], key=lambda x: x[0])
    bottom = sorted(points[2:], key=lambda x: x[0], reverse=True)
    points = np.array(top + bottom, dtype="float32")

    # 幅と高さを計算
    width = max(
        np.sqrt(((points[0][0] - points[2][0]) ** 2) * 2),
        np.sqrt(((points[1][0] - points[3][0]) ** 2) * 2),
    )
    height = max(
        np.sqrt(((points[0][1] - points[2][1]) ** 2) * 2),
        np.sqrt(((points[1][1] - points[3][1]) ** 2) * 2),
    )

    # 変換先のポイントを定義
    dst = np.array(
        [
            np.array([0, 0]),
            np.array([width - 1, 0]),
            np.array([width - 1, height - 1]),
            np.array([0, height - 1]),
        ],
        np.float32,
    )

    # 透視変換行列を計算
    trans = cv2.getPerspectiveTransform(points, dst)
    # 透視変換を適用
    return cv2.warpPerspective(img, trans, (int(width), int(height)))


def clippingIMG(img):
    """画像から四角形領域を抽出して保存する関数

    Args:
        img (ndarray): 入力画像
        outputPath (str, optional): 出力先パス. デフォルトは './img/output.png'.
    """
    # 入力画像をコピーして輪郭描画用の画像を作成
    lines = img.copy()

    # 画像をグレースケールに変換
    canny = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 画像をガウシアンブラーで平滑化
    canny = cv2.GaussianBlur(canny, (5, 5), 0)

    # Cannyアルゴリズムでエッジ検出
    canny = cv2.Canny(canny, 50, 100)

    # エッジ検出画像から輪郭を抽出
    contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 輪郭を面積の大きい順にソート
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    warp = None  # 最適な四角形の輪郭を保持する変数

    # 各輪郭を処理
    for i, c in enumerate(contours):
        # 輪郭の周長を計算
        arclen = cv2.arcLength(c, True)

        # 輪郭をポリゴンに近似
        approx = cv2.approxPolyDP(c, 0.02 * arclen, True)

        # 輪郭のレベルを計算（色の強度に使用）
        level = 1 - float(i) / len(contours)

        # 近似した輪郭が四角形かどうかを確認
        if len(approx) == 4:
            # 四角形の輪郭を画像に描画
            cv2.drawContours(lines, [approx], -1, (0, 0, 255 * level), 2)

            # 最初に見つかった四角形の輪郭を保存
            if warp is None:
                warp = approx.copy()
        else:
            # 四角形でない輪郭を画像に描画
            cv2.drawContours(lines, [approx], -1, (0, 255 * level, 0), 2)

        # 輪郭の頂点に円を描画
        for pos in approx:
            cv2.circle(lines, tuple(pos[0]), 4, (255 * level, 0, 0))

    # 四角形の輪郭が見つかった場合
    if warp is not None:
        print(warp)
        # 四角形の輪郭を使って画像を変換
        warped = transform_by4(img, warp[:, 0, :])
    
    return warped


def camera(cap, res_dir="test/"):
    """カメラから画像をキャプチャする関数

    Args:
        cap : カメラ
        res_dir (str, optional): 出力先パス. デフォルトは 'test/'

    Returns:
        frame: 撮った写真
    """
    frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("fish-camera", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            cv2.imwrite(res_dir, frame)
            cv2.destroyAllWindows()
            break
    return frame


def camera_start():
    """カメラを起動し、画像をキャプチャしてクリッピングを行う関数"""
    res_dir = "./output/camera/"
    os.makedirs(res_dir, exist_ok=True)

    img = None  # 初期化

    # カメラインデックスを試行する
    for index in range(5):
        cap = cv2.VideoCapture(index)
        
        # 解像度の設定
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        
        # カメラが使用できれば撮影する
        if cap.isOpened():
            print(f"Camera {index} is opened")
            img = camera(cap, res_dir + 'fish.png')
            cap.release()  # カメラを解放
            break
        else:
            print(f"Camera {index} failed to open")

    # カメラで画像が撮影できなかった場合、エラーメッセージを表示して終了
    if img is None:
        print("画像のキャプチャに失敗しました")
        return

    # クリッピング
    clip_img = clippingIMG(img)
    cv2.imwrite(res_dir+'clip.png', clip_img)
    
    # 影を除去
    rsimg = removeShade(clip_img)
    cv2.imwrite(res_dir+'rshade.png', rsimg)

    # 結果をinputへ書き込む
    filename = module.namingFile("fish_data",".png","./inputimg")
    filepath = "./inputimg/"+filename
    cv2.imwrite(filepath,clip_img)

if __name__ == "__main__":
    print('main.pyを実行してください')