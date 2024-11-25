import cv2
import numpy as np
import open3d as o3d
import os
from scipy.spatial import KDTree



def calcCenterX(points_3d_base):
    """
    y座標ごとのx座標の平均を計算する関数

    Args:
        points_3d_base: 3Dポイントのリストまたはnumpy配列

    Returns:
        各y座標に対応するx座標の平均のリスト
    """
    # 入力をnumpy配列に変換
    points_3d_base = np.array(points_3d_base)

    # ユニークなy座標のリストを取得
    unique_y = np.unique(points_3d_base[:, 1])

    # 各ユニークなy座標に対して、対応するx座標の平均を計算
    result = [points_3d_base[points_3d_base[:, 1] == y][:, 0].mean() for y in unique_y]

    return result

def createPointCloudFromIMG(image, height, maskimg):
    """画像から3D点群を生成する

    Args:
        image : 入力画像
        height : 押し出しの高さ
        mask : マスク
    """

    # 出力結果と画像の向きを一緒にさせるため上下反転
    image = cv2.flip(image, 0)
    h, w = image.shape

    # 2次元の点群を生成
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    z0 = np.zeros_like(x, dtype=np.float32)

    # マスクを適用して点群をフィルタリング
    mask = image > 0
    points_3d_base = np.stack((x[mask], y[mask], z0[mask]), axis=-1)

    # 中心地を計算
    center_x = np.mean(points_3d_base[:, 0])
    center_y = np.mean(points_3d_base[:, 1])
    center_xPerY = calcCenterX(points_3d_base)

    # 1画素あたりの高さを算出
    # 初期値はheight/2の高さでcenterまで高くなる
    heightPixcel = heightPixcel_x = (height / 2) / (
        center_x - points_3d_base[:, 0].min()
    )
    heightPixcel_y = (height / 2) / (center_y - points_3d_base[:, 1].min())

    # 上面と底面を生成
    base_points = []
    top_points = []

    # 初期値の設定
    top_he = base_he = height / 2
    bf_x = bf_y = i = 0

    for point in points_3d_base:
        # x,y座標を取得
        x, y, _ = point

        # y座標が変わっていたら初期化
        if bf_y != y:
            top_he = base_he = height / 2
            i += 1
            # yの中心地より上なら
            if center_y > y:
                heightPixcel_y += heightPixcel_y
            else:
                heightPixcel_y -= heightPixcel_y
                if heightPixcel_y < height / 2:
                    heightPixcel_y = height / 2
            bf_y = y

        # iが範囲外に出ないようにする
        if i >= len(center_xPerY):
            i = len(center_xPerY) - 1

        # 中心まで
        if x < center_xPerY[i]:
            # 高さの上限まで到達していなければ
            if top_he < height / 2 + heightPixcel_y:
                top_he += heightPixcel
                base_he -= heightPixcel
            # 到達していたら
            else:
                top_he = height
                base_he = 0
        # 中心以降
        else:
            top_he -= heightPixcel
            base_he += heightPixcel
            if top_he < base_he:
                top_he = base_he = height / 2

        # 条件に一致するとリストに加える
        if x % 10 == 0 and y % 10 == 0:
            base_points.append([x, y, base_he])
            top_points.append([x, y, top_he])

    # 輪郭の抽出
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    contour_points = []
    for contour in contours:
        contour = contour.squeeze()
        if contour.ndim == 2:
            contour_points.append(contour)
    # 側面の点群生成
    side_points = []
    for contour in contour_points:
        for point in contour:
            x, y = point
            if x % 1 == 0 and y % 1 == 0: # 調整しやすいようこの組み方を…
                side_points.append([x, y, height / 2])
    
    return base_points,top_points,side_points


def printAry(mesh):
    """meshを出力するだけ

    Args:
        mesh : mesh
    """
    # メッシュ情報のNumPy配列への変換
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    vertex_normals = np.asarray(mesh.vertex_normals)
    triangle_normals = np.asarray(mesh.triangle_normals)
    vertex_colors = np.asarray(mesh.vertex_colors)
    triangle_uvs = np.asarray(mesh.triangle_uvs)

    # メッシュ情報の表示
    print("Vertices:")
    print(vertex_colors)


def estimate_radius(points, k=8):
    """
    点群の近傍距離に基づいて適切な半径を推定

    Args:
        points (np.ndarray): Nx3の点群データ
        k (int): 最近傍点の数

    Returns:
        float: 推定された適切な半径
    """
    tree = KDTree(points)
    distances, _ = tree.query(points, k=k+1)  # 自身を含むためk+1
    mean_distance = np.mean(distances[:, 1:])  # 自身を除外
    return mean_distance

def createMesh(points, radii=None):
    """
    点群からBall Pivoting Algorithm (BPA) を使用してメッシュを生成

    Args:
        points (np.ndarray): 点群データ (Nx3 の NumPy 配列)
        radii (list[float]): BPAに使用する半径リスト (Noneなら自動推定)

    Returns:
        o3d.geometry.TriangleMesh: 生成されたメッシュ
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # 法線推定
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    point_cloud.orient_normals_consistent_tangent_plane(10)
    
    # 半径を自動推定
    if radii is None:
        estimated_radius = estimate_radius(points)
        radii = [estimated_radius * i for i in np.linspace(0.01, 3.0, 10)]
    
    # BPAでメッシュ生成
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        point_cloud,
        o3d.utility.DoubleVector(radii)
    )

    print(radii)

    # 後処理
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    
    return mesh


def saveOBJFile(filePath, vertices, uvs, normals, faceVertIDs, uvIDs, normalIDs, vertexColors):
    """OBJを出力する

    Args:
        filePath : ファイルパス
        vertices : 頂点
        uvs : UV
        normals : 法線
        faceVertIDs : 面情報
        uvIDs : UV
        normalIDs : 法線ID
        vertexColors : 頂点の色
    """
    file = open(filePath, 'w')
    
    # 設定文章
    file.write("######\n")
    file.write("#\n")
    file.write("# Made by TomitaAkito\n")
    file.write("# For Resurch 2024-2025\n")
    file.write("# Info: VisualComputingLab.\n")
    file.write("#\n")
    file.write("######\n")
    
    #? 頂点と色情報(v)
    for vi,v in enumerate(vertices):
        # 頂点
        vStr = "v %s %s %s"  %(v[0], v[1], v[2])
        # 色情報
        if len( vertexColors) > 0:
            color = vertexColors[vi]
            vStr += " %s %s %s\n" %(color[0], color[1], color[2])
        else:
            vStr += " 0 0 0\n"
            
        file.write(vStr)
    file.write("# %s vertices\n\n"  %(len(vertices)))
    
    #? uv展開(vt)
    for uv in uvs:
        uvStr =  "vt %s %s\n"  %(uv[0], uv[1])
        file.write(uvStr)
    file.write("# %s uvs\n\n"  %(len(uvs)))
    
    #? 法線(vn)
    for n in normals:
        nStr =  "vn %s %s %s\n"  %(n[0], n[1], n[2])
        file.write(nStr)
    file.write("# %s normals\n\n"  %(len(normals)))
    
    #? 面情報(f)
    for fi, fvID in enumerate(faceVertIDs):
        fStr = "f"
        for fvi, fvIDi in enumerate( fvID ):
            fStr += " %s" %( fvIDi + 1 )

            if len(uvIDs) > 0:
                fStr += "/%s" %( fvIDi + 1  )
            if len(normalIDs) > 0:
                fStr += "/%s" %( normalIDs[fi][fvi] + 1 )

        fStr += "\n"
        file.write(fStr)
    file.write("# %s faces\n\n"  %( len( faceVertIDs)) )

    file.write("# End of File\n")
    file.close()

def disassemblyMesh(mesh):
    """meshから以下の要素を分解して返す
    頂点・UV・法線・表面ID・UVID・法線ID・頂点の色

    Args:
        mesh : メッシュ

    Returns:
        numpy型の先述した要素
    """
    
    # 頂点
    vertices = np.asarray(mesh.vertices)
    # UV
    uvs = np.asarray(mesh.triangle_uvs)
    # 法線
    normals = np.asarray(mesh.triangle_normals)
    # 表面ID
    faceVertIDs = np.asarray(mesh.triangles)
    # UVID
    uvIDs = np.empty(0)
    # 法線ID
    normalIDs = np.empty(0)
    # 頂点の色
    vertexColors = np.array([[0, 0, 0]] * len(vertices))
    
    # 頂点数と頂点色の次元数が一致するかを確認
    assert vertices.shape[0] == vertexColors.shape[0], "Vertices and vertexColors length do not match"
    
    return vertices, uvs, normals, faceVertIDs, uvIDs, normalIDs, vertexColors

def computeNormals(vertices, faces):
    """
    頂点の法線ベクトルを計算する関数

    Args:
        vertices : 頂点座標のリストまたはnumpy配列
        faces : 三角形メッシュの面情報（頂点インデックスのリスト）

    Returns:
        各頂点の法線ベクトルのリスト
    """
    # 頂点数と同じ長さのゼロベクトルを初期化
    normals = np.zeros(vertices.shape, dtype=np.float32)
    
    # 各三角形の法線を計算して加算
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        
        # 三角形の辺ベクトルを計算
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # 外積を計算して三角形の法線を得る
        face_normal = np.cross(edge1, edge2)
        
        # 各頂点に法線を加算
        normals[face[0]] += face_normal
        normals[face[1]] += face_normal
        normals[face[2]] += face_normal

    # 正規化して各頂点の法線を求める
    norm = np.linalg.norm(normals, axis=1)
    norm[norm == 0] = 1  # ゼロ除算を防ぐために0の要素を1に置換
    normals /= norm[:, np.newaxis]

    return normals


def make_normals_outward(mesh):
    """
    すべての法線ベクトルを外向きに修正する関数。

    Args:
        mesh: Open3Dのメッシュオブジェクト

    Returns:
        法線ベクトルが外向きに修正されたメッシュオブジェクト
    """
    # メッシュの頂点座標を取得
    vertices = np.asarray(mesh.vertices)
    # メッシュの三角形（面）のインデックスを取得
    triangles = np.asarray(mesh.triangles)
    
    # メッシュの重心を計算
    mesh_center = vertices.mean(axis=0)
    
    # 三角形の法線を計算（法線が未計算の場合、これを実行）
    mesh.compute_triangle_normals()
    triangle_normals = np.asarray(mesh.triangle_normals)
    
    # 新しい三角形リストを作成（反転したものを保存）
    for i, tri in enumerate(triangles):
        # 各三角形の頂点の座標
        tri_vertices = vertices[tri]
        
        # 三角形の重心を計算
        tri_center = np.mean(tri_vertices, axis=0)
        
        # 三角形の重心からメッシュの重心に向かうベクトル
        center_to_face = tri_center - mesh_center
        
        # 三角形の法線と重心からメッシュの重心へのベクトルが内積が負なら、法線が内側を向いている
        if np.dot(triangle_normals[i], center_to_face) < 0:
            # 法線が内向きの場合、三角形の頂点順序を反転して法線を修正
            triangles[i] = triangles[i][::-1]
    
    # 修正した三角形のインデックスを更新
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # 法線を再計算
    mesh.compute_vertex_normals()
    
    return mesh

def showPoint(points_3d):
    
    # NumPy配列からOpen3Dの点群オブジェクトを作成
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3d)

    # 点群の法線を計算
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
    )

    # 点群を可視化
    o3d.visualization.draw_geometries([point_cloud])
    
def save_ply(filename, points):
    """点群をPLYファイルとして保存する

    Args:
        filename (str): 保存先のファイル名
        points (numpy.ndarray): 点群（N x 3 の形式）
    """
    with open(filename, 'w') as f:
        # ヘッダーを書く
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")

        # 点群データを書く
        for point in points:
            f.write(f"{point[0]} {point[1]} {point[2]}\n")


def creating3D(filePath, maskPath, filename, height=100, smoothFlag=False):
    """3Dオブジェクトを生成する

    Args:
        filePath : ファイルのパス
        maskPath : マスク画像のパス
        file : ファイルの名前
        height : 押し出しの高さ
        smoothFlag : 平滑化するか
    """
    # 拡張子を除くファイル名を作る
    exFile = os.path.splitext(filename)[0]

    # 画像を読み込む(グレースケール)
    img = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)

    # ノイズ除去（メディアンフィルタ）
    img = cv2.medianBlur(img, 5)

    # 二値化処理
    _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

    # 画像全体から3次元点群を生成
    base_points,top_points,side_points = createPointCloudFromIMG(thresh, height, mask)
    
    # 3次元点群を連結
    # base_points = np.vstack((base_points, top_points, side_points))
    # base_points = np.vstack((base_points, side_points))
    top_points = np.vstack((top_points, side_points))

    # 点群の出力
    # ply_path = f"./output/base_points_{exFile}.ply"
    # save_ply(ply_path, np.array(base_points))
    # print(f"PLYファイルとして保存されました: {ply_path}")

    # 以後の処理のため、numpy配列に変更
    base_points = np.array(base_points)
    top_points = np.array(top_points)
    
    # 点群を表示
    showPoint(base_points)
    showPoint(top_points)
    
    # メッシュを生成
    base_mesh = createMesh(base_points)
    top_mesh = createMesh(top_points)
    mesh = base_mesh + top_mesh
    
    # すべての法線ベクトルを外向きに修正
    mesh = make_normals_outward(mesh)
    
    # Meshを表示
    o3d.visualization.draw_geometries([mesh])

    # OBJとして出力
    vertices, uvs, normals, faceVertIDs, uvIDs, normalIDs, vertexColors = disassemblyMesh(mesh)

    # 法線ベクトルを手動で計算
    normals = computeNormals(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    
    meshpath = "./output/mesh/me_" + exFile + ".obj"
    saveOBJFile(meshpath, vertices, uvs, normals, faceVertIDs, uvIDs, normalIDs, vertexColors)

if __name__ == "__main__":
    print('main.pyを実行してください')