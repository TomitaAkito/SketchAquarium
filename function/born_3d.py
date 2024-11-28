import numpy as np
import trimesh
import os

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
            f.write(f"Bone: {child_name}, World Position: {world_position.tolist()}, Parent: {parent_name}\n")

    # メッシュにスキンウェイトを適用 (擬似的に頂点カラーとしてボーンの影響範囲を適用)
    vertices = mesh.vertices
    weights = np.zeros((len(vertices), len(bone_data)))
    
    for i, (parent, child) in enumerate(bone_data):
        direction = child - parent
        for j, vertex in enumerate(vertices):
            vec_to_bone = vertex - parent
            projection = np.dot(vec_to_bone, direction) / np.linalg.norm(direction)
            weights[j, i] = max(0, 1 - abs(projection))  # ボーンへの距離でスキンウェイトを設定
    
    # 頂点カラーとしてウェイトを設定 (擬似的に視覚化)
    mesh.visual.vertex_colors = (weights.max(axis=1) * 255).astype(np.uint8)

    # メッシュを保存
    output_obj_path = f"{file_name}.obj"
    mesh.export(output_obj_path)

    print(f"ボーン情報を保存しました: {txt_path}")
    print(f"スキンウェイト適用メッシュを保存しました: {output_obj_path}")
    visualize_with_open3d(mesh,weights)
    
import open3d as o3d
import matplotlib.pyplot as plt

# Open3Dを使ってメッシュを可視化
def visualize_with_open3d(mesh, weights):
    # 頂点カラーをスケール補正
    scale_factor = 5  # 明るさ補正係数（必要に応じて調整）
    vertex_colors = weights.max(axis=1)  # 各頂点の最大ウェイトを使用
    vertex_colors_scaled = (vertex_colors * scale_factor).clip(0, 1)  # 明るさ補強後0-1に正規化

    # カラーマップを適用
    cmap = plt.cm.viridis  # 他のカラーマップも利用可能
    vertex_colors_rgb = cmap(vertex_colors_scaled)[:, :3]  # RGB成分を取得

    # Open3D用のメッシュを構築
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors_rgb)

    # メッシュを可視化
    o3d.visualization.draw_geometries([o3d_mesh], window_name="Weight Visualization",
                                      width=800, height=600)

def main(obj_path,file_name):
    # サンプルデータ
    bone_list = [
        [[0, 0, 0], [1, 1, 1], [2, 2, 0]],
        [[1, 1, 1], [1, 2, 2], [0, 3, 1]]
    ]

    # 実行
    create_bone_structure(bone_list, obj_path, file_name)

if __name__ == "__main__":    
    main("output/mesh/me_fish_data(1).obj","output_bone_mesh")