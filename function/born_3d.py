import pywavefront
import numpy as np
import open3d as o3d

# OBJファイルを読み込み
def load_obj(file_path):
    scene = pywavefront.Wavefront(file_path, collect_faces=True)
    vertices = np.array(scene.vertices)
    faces = np.array([face for face in scene.mesh_list[0].faces])
    
    # Open3D用のメッシュに変換
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    
    return mesh

# ボーンデータを手動で定義
class Bone:
    def __init__(self, name, position, parent=None):
        self.name = name
        self.position = np.array(position)
        self.parent = parent
        self.offset = np.array([0, 0, 0])  # ボーンの移動によるオフセット

    def get_world_position(self):
        if self.parent:
            return self.parent.get_world_position() + self.position
        return self.position

    def move(self, offset):
        # ボーンを指定されたオフセットで移動
        self.offset = np.array(offset)

# ボーンリストを作成
def create_bones():
    root_bone = Bone("root", [300, 300, 50])
    bone1 = Bone("bone1", [150, 0, 0], root_bone)
    bone2 = Bone("bone2", [50, 0, 0], bone1)
    return [root_bone, bone1, bone2]

# ボーン情報を表示
def display_bones(bones):
    for bone in bones:
        world_position = bone.get_world_position()
        parent_name = bone.parent.name if bone.parent else "None"
        print(f"Bone: {bone.name}, World Position: {world_position}, Parent: {parent_name}")

# ボーン情報をファイルに保存
def save_bone_info(bones, file_path):
    with open(file_path, 'w') as f:
        for bone in bones:
            world_position = bone.get_world_position()
            parent_name = bone.parent.name if bone.parent else "None"
            f.write(f"Bone: {bone.name}, World Position: {world_position.tolist()}, Parent: {parent_name}\n")

# 各頂点に対するボーンの影響（スキンウェイト）を設定
class VertexWeight:
    def __init__(self, bone, weight):
        self.bone = bone
        self.weight = weight

# 頂点にボーンウェイトを割り当てる
def assign_vertex_weights(mesh, bones):
    vertex_weights = []
    for vertex in mesh.vertices:
        # 簡単な例: すべての頂点がbone1の影響を100%受ける
        weight = VertexWeight(bones[1], 1.0)
        vertex_weights.append(weight)
    return vertex_weights

# ボーンの動きに基づいてメッシュを変形
def deform_mesh(mesh, bones, vertex_weights):
    new_vertices = []
    for i, vertex in enumerate(mesh.vertices):
        weight = vertex_weights[i]
        bone = weight.bone
        
        # ボーンの影響に基づいて頂点を変形
        new_vertex = vertex + weight.weight * (bone.get_world_position() + bone.offset - bone.position)
        new_vertices.append(new_vertex)
    
    # メッシュの頂点位置を更新
    mesh.vertices = o3d.utility.Vector3dVector(new_vertices)

# メイン処理
if __name__ == "__main__":

    # OBJファイルのパスを指定
    obj_path = 'test.obj'
    
    # OBJファイルの読み込み
    mesh = load_obj(obj_path)
    
    # ボーンの作成
    bones = create_bones()
    
    # 各頂点に対するボーンの影響を設定
    vertex_weights = assign_vertex_weights(mesh, bones)
    
    # ボーンの表示
    display_bones(bones)
    
    # ボーン情報をファイルに保存
    save_bone_info(bones, 'bone_info.txt')
    
    # ボーンを動かしてみる（例としてbone1を移動）
    bones[1].move([0.5, 0.5, 0.0])  # ボーン1を少し移動
    
    # ボーンの移動に基づいてメッシュを変形
    deform_mesh(mesh, bones, vertex_weights)
    
    # OBJ形式でメッシュをエクスポート
    o3d.io.write_triangle_mesh("output_model.obj", mesh, write_ascii=True)

    print("メッシュとボーン情報を出力しました。")
