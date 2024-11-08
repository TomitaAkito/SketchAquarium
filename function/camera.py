import cv2
import numpy as np
import os
import sys
# 現在のディレクトリをシステムパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import module


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
    res_dir = "./output/camera/"
    os.makedirs(res_dir, exist_ok=True)

    img = None

    for index in range(5):
        cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        
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

    clip_img = clippingIMG(img)
    if clip_img is not None:
        cv2.imwrite(res_dir+'clip.png', clip_img)
        
        rsimg = removeShade(clip_img)
        cv2.imwrite(res_dir+'rshade.png', rsimg)

        filename = module.namingFile("fish_data",".png","./inputimg")
        filepath = "./inputimg/"+filename
        cv2.imwrite(filepath,rsimg)
    else:
        print("クリッピングに失敗しました")


if __name__ == "__main__":
    print('main.pyを実行してください')
