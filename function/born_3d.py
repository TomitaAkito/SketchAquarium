import pywavefront
import pythreejs as three
import numpy as np


def load_obj(mesh_path):
    scene = pywavefront.Wavefront(mesh_path, collect_faces=True)
    return scene


def create_mesh(scene):
    vertices = np.array(scene.vertices, dtype=np.float32)
    faces = [item for sublist in scene.mesh_list[0].faces for item in sublist]
    faces = np.array(faces, dtype=np.uint32).ravel()  # インデックスをフラットにする

    geometry = three.BufferGeometry(
        attributes={
            "position": three.BufferAttribute(vertices, normalized=False),
            "index": three.BufferAttribute(faces, normalized=False),
        }
    )

    material = three.MeshStandardMaterial(
        color="#808080", wireframe=True
    )  # 'grey' を '#808080' に変更
    mesh = three.Mesh(geometry=geometry, material=material)
    return mesh


def create_armature(bone_points):
    bones = []
    for i, point in enumerate(bone_points):
        bone = three.Bone(position=point)
        if i > 0:
            bones[i - 1].add(bone)
        bones.append(bone)
    return bones[0]  # root bone


def save_scene(scene, output_path):
    with open(output_path, "w") as file:
        file.write(scene)


if __name__ == "__main__":
    mesh_path = "./output/mesh/me_fish_data(1).obj"
    output_path = "./output/mesh/me_fish_data(12).json"

    bone_points = [
        [0, 0, 0],  # ボーンのポイント1
        [50, 50, 50],  # ボーンのポイント2
        [100, 100, 100],  # ボーンのポイント3
    ]

    # メッシュの読み込み
    wavefront_scene = load_obj(mesh_path)

    # メッシュの作成
    mesh = create_mesh(wavefront_scene)

    # アーマチュア（ボーン）の作成
    armature = create_armature(bone_points)

    # シーンの作成
    scene = three.Scene(children=[mesh, armature])

    # シーンを保存
    save_scene(scene.to_json(), output_path)
