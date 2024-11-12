import cv2
import numpy as np
import os
import sys
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
# 現在のディレクトリをシステムパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import module


g_cameraFlag = False # 撮ってきた写真が問題ないか判定するフラグ

def removeShade(img):
    """画像から影を除去する"""
    rgb_planes = cv2.split(img)
    result_planes = []

    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        result_planes.append(diff_img)

    im_shadow = cv2.merge(result_planes)
    return im_shadow


def transform_by4(img, points, offset=5):
    """画像を透視変換を用いて切り抜く"""
    points = sorted(points, key=lambda x: x[1])
    top = sorted(points[:2], key=lambda x: x[0])
    bottom = sorted(points[2:], key=lambda x: x[0], reverse=True)
    points = np.array(top + bottom, dtype="float32")

    # 各点を内側に少しオフセットして黒枠が映らないように調整
    points[0] += [offset, offset]
    points[1] += [-offset, offset]
    points[2] += [-offset, -offset]
    points[3] += [offset, -offset]

    width = max(
        np.linalg.norm(points[0] - points[1]),
        np.linalg.norm(points[2] - points[3])
    )
    height = max(
        np.linalg.norm(points[0] - points[3]),
        np.linalg.norm(points[1] - points[2])
    )

    dst = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ],
        dtype="float32"
    )

    trans = cv2.getPerspectiveTransform(points, dst)
    return cv2.warpPerspective(img, trans, (int(width), int(height)))


def clippingIMG(img):
    """画像から最大の四角形領域を抽出して保存する関数"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 100)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    warp = None

    for c in contours:
        arclen = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * arclen, True)

        if len(approx) == 4:
            warp = approx
            break

    if warp is not None:
        warped = transform_by4(img, warp[:, 0, :])
        return warped
    else:
        print("四角形の輪郭が見つかりませんでした")
        return None
    
def sharpen_image(image):
    """
    画像を鮮鋭化する関数。
    Args:
        image: 入力画像 (NumPy配列)
    Returns:
        鮮鋭化された画像
    """
    # 鮮鋭化のためのカーネル
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    
    # フィルタを適用して鮮鋭化
    sharpened_image = cv2.filter2D(image, -1, kernel)
    
    return sharpened_image

def display_image(image_path,title="画像選択" ,flag=True,root=None):
    """imgを表示する

    Args:
        image_path (str): 画像のパス
        root :メッセージボックス
    """
    global g_cameraFlag
    
    
    if root is None:
        root = tk.Tk()  # 新しいTkウィンドウの代わりにToplevelを使用
    else:
        root = tk.Toplevel(root)  # 既存のTkインスタンスを親にする

    root.title(title)

    # 画像を開く
    img = Image.open(image_path)
    img = img.resize((300, 300), Image.Resampling.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)

    # 画像を表示
    img_label = tk.Label(root, image=img_tk)
    img_label.image = img_tk  # 参照を保持するために属性に保存
    img_label.pack()
    
    # Yes/Noボタン
    yes_button = tk.Button(root, text="Yes", command=lambda: displat_quit(root,True))
    yes_button.pack(side=tk.LEFT, padx=10, pady=10)

    no_button = tk.Button(root, text="No", command=lambda: displat_quit(root))  # Noボタンには何も追加処理を行わない
    no_button.pack(side=tk.RIGHT, padx=10, pady=10)

    root.mainloop()
    
    
def displat_quit(root,yes_flag = False):
    """ディスプレイをすべて閉じる

    Args:
        root : 親のtkインスタンス
        yes_flag : yesかどうかを判定
    """
    global g_cameraFlag
    
    root.quit() 
    root.destroy()
    if yes_flag:
        g_cameraFlag = True

def regitFish():
        """カメラ画像の魚を生成アルゴリズムに反映させる
        """
        # ライブラリを読み込む
        import function.extraction_fish as extraction_fish
        
        # file名を取得する
        file = "camera.png"
        filePath = "./output/camera/" + file

        # 魚を抽出
        only_img = extraction_fish.findFishParts(filePath, file)

def camera(cap, res_dir="test/"):
    """カメラから画像をキャプチャする関数"""
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
    global g_cameraFlag
    
    res_dir = "./output/camera/"

    img = None
    g_cameraFlag = False
    
    while True:
        
        # cameraを開く
        for index in range(5):
            # cameraの初期設定
            cap = cv2.VideoCapture(index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
            
            # cameraが開いたら画像を取得する処理へ
            if cap.isOpened():
                print(f"Camera {index} is opened")
                img = camera(cap, res_dir + 'fish.png')
                cap.release()
                break
            else:
                print(f"Camera {index} failed to open")

        if img is None:
            print("画像のキャプチャに失敗しました")
            return

        # クリッピング
        clip_img = clippingIMG(img)
        if clip_img is not None:
            cv2.imwrite(res_dir+'clip.png', clip_img)
            
            # 影処理
            resultImg = rsimg = removeShade(clip_img)
            cv2.imwrite(res_dir+'rshade.png', rsimg)
            
            # 鮮鋭化
            # resultImg = sharpImg = sharpen_image(rsimg)
            # cv2.imwrite(res_dir+'sharp.png', sharpImg)

            # 利用者に問題ないか尋ねる
            cv2.imwrite(res_dir+'camera.png', resultImg)
            regitFish()
            display_image('./output/onlyfish/onlycamera.png')
            
            if g_cameraFlag:
                # 写真を入力として保存
                filename = module.namingFile("fish_data",".png","./inputimg")
                filepath = "./inputimg/"+filename
                cv2.imwrite(filepath,resultImg)
                
                return
            
        else:
            print("クリッピングに失敗しました")


if __name__ == "__main__":
    print('main.pyを実行してください')
