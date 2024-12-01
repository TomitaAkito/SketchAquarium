import numpy as np
import trimesh
import open3d as o3d
import matplotlib.pyplot as plt
import cv2


def create_bone_structure(bone_list, obj_path, file_name):
    """
    ボーン構築とメッシュのスキンウェイトを施し、メッシュと情報を保存する。

    Args:
        bone_list (list): ボーンリスト。形式: [[親[x, y, z], 子1[x1, y1, z1], 子2[x2, y2, z2], ...], ...]
        obj_path (str): メッシュのパス。
        file_name (str): 保存するファイル名（拡張子なし）。
    """
    # メッシュをロード
    mesh = trimesh.load_mesh(obj_path)

    # ボーンデータを構築
    bone_data = []
    for bone in bone_list:
        parent_position = np.array(bone[0])
        for child_position in bone[1:]:
            child_position = np.array(child_position)
            bone_data.append((parent_position, child_position))

    # ボーン情報をtxtに保存
    txt_path = f"{file_name}.txt"
    with open(txt_path, "w") as f:
        for i, (parent, child) in enumerate(bone_data):
            parent_name = f"Bone_{i}_Parent"
            child_name = f"Bone_{i}_Child"
            world_position = child
            f.write(
                f"Bone: {child_name}, World Position: {world_position.tolist()}, Parent: {parent_name}\n"
            )

    # メッシュにスキンウェイトを適用 (擬似的に頂点カラーとしてボーンの影響範囲を適用)
    vertices = mesh.vertices
    weights = np.zeros((len(vertices), len(bone_data)))

    for i, (parent, child) in enumerate(bone_data):
        direction = child - parent
        for j, vertex in enumerate(vertices):
            vec_to_bone = vertex - parent
            projection = np.dot(vec_to_bone, direction) / np.linalg.norm(direction)
            weights[j, i] = max(
                0, 1 - abs(projection)
            )  # ボーンへの距離でスキンウェイトを設定

    # 頂点カラーとしてウェイトを設定 (擬似的に視覚化)
    mesh.visual.vertex_colors = (weights.max(axis=1) * 255).astype(np.uint8)

    # メッシュを保存
    output_obj_path = f"{file_name}.obj"
    mesh.export(output_obj_path)

    print(f"ボーン情報を保存しました: {txt_path}")
    print(f"スキンウェイト適用メッシュを保存しました: {output_obj_path}")
    visualize_with_open3d(mesh, weights)


def visualize_with_open3d(mesh, weights):
    """Open3Dを使ってメッシュを可視化

    Args:
        mesh (o3d): メッシュ
        weights : ウェイト
    """
    # 頂点カラーをスケール補正
    scale_factor = 5  # 明るさ補正係数（必要に応じて調整）
    vertex_colors = weights.max(axis=1)  # 各頂点の最大ウェイトを使用
    vertex_colors_scaled = (vertex_colors * scale_factor).clip(
        0, 1
    )  # 明るさ補強後0-1に正規化

    # カラーマップを適用
    cmap = plt.cm.viridis  # 他のカラーマップも利用可能
    vertex_colors_rgb = cmap(vertex_colors_scaled)[:, :3]  # RGB成分を取得

    # Open3D用のメッシュを構築
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors_rgb)

    # メッシュを可視化
    o3d.visualization.draw_geometries(
        [o3d_mesh], window_name="Weight Visualization", width=800, height=600
    )
    
    
def calculate_centroids(image_path, radius=10, showFlag=False):
    """
    二値化された画像において、各白領域の最初の重心を計算する関数。
    
    Args:
        image_path (str): 二値化画像のファイルパス
    
    Returns:
        list: 最初の重心の座標 [x1, y1] 形式
    """
    # 画像を読み込む
    binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if binary_image is None:
        raise ValueError(f"画像の読み込みに失敗しました: {image_path}")
    
    # 画像が二値化されていない場合、二値化処理を行う
    _, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

    # 連結成分をラベリング
    num_labels, labels = cv2.connectedComponents(binary_image)

    # 画像をカラー画像に変換（赤色を表示するため）
    color_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    
    for label in range(1, num_labels):  # 0は背景なので除外
        # ラベルごとの座標を抽出
        coords = np.column_stack(np.where(labels == label))
        
        # 最初の重心を計算
        centroid = np.mean(coords, axis=0)
        x, y = centroid[1], centroid[0]  # [x, y] 形式に変換

        # 重心を中心に円を描く
        cv2.circle(color_image, (int(x), int(y)), radius, (0, 0, 255), 2)  # 赤色の円を描く
        
        if showFlag:
            # 画像を表示
            cv2.imshow("Marked Image", color_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return [x, y]  # 最初の重心を返す

    # 重心が見つからない場合、デフォルト値を返す
    return [0, 0]


    
    
def MakeBornList(file_name, mask_num, mask_directory="output/ss/", height=100):
    """関節の数を基にボーンリストを作成

    Args:
        file_name: 拡張子なしのファイル名
        mask_num: マスクの数
        mask_directory: マスクが格納されているディレクトリ. Defaults to "output/ss/".
        height: 魚が生成される高さ. Defaults to 100.

    Returns:
        born_list: ボーンの情報(全体[1stボーン[親[],関節[],部位[]],2ndボーン[...]])
    """
    def add_z_coordinate(point, z_value):
        """[x, y]形式のポイントにz座標を追加して[x, y, z]形式に変換"""
        if not point or len(point) < 2:
            print(f"警告: 無効なポイント {point} を検出しました。デフォルト値を使用します。")
            return [0, 0, z_value]  # デフォルト値
        return [point[0], point[1], z_value]

    born_list = []
    z_value = height / 2

    # 最も大きい部分(親・胴体)のポイント
    mask_path = mask_directory + "2_regrow_" + file_name + "_" + str(mask_num - 1) + ".png"
    body_point = add_z_coordinate(calculate_centroids(mask_path), z_value)

    # 例外処理
    if mask_num <= 0:
        return [body_point]

    # マスクがなくなるまで
    for i in range(0, mask_num):
        # 重複部分(関節)のポイント
        mask_path = mask_directory + "2_regrow" + file_name + "_common_" + str(i) + ".png"
        joint_point = add_z_coordinate(calculate_centroids(mask_path), z_value)

        # 部位部分のポイント
        mask_path = mask_directory + "2_regrow" + file_name + "_small_" + str(i) + ".png"
        part_point = add_z_coordinate(calculate_centroids(mask_path), z_value)

        # 全体[1stボーン[親[],関節[],部位[]],2ndボーン[...]] になるようリストを作成
        # 〇番目のボーン
        born_list.append([body_point, joint_point, part_point])
    
    print(born_list)
    return born_list



def main(obj_path, file_name):
    # サンプルデータ
    bone_list = [
        [[0, 0, 0], [1, 1, 1], [2, 2, 0]],
        [[1, 1, 1], [1, 2, 2], [0, 3, 1]],
        [[0, 0, 0], [2, 2, 2], [9, 9, 0]],
    ]

    # 実行
    create_bone_structure(bone_list, obj_path, file_name)


if __name__ == "__main__":
    main("output/mesh/me_template(1).obj", "output_bone_mesh")
