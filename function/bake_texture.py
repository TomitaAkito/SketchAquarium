import numpy as np
from PIL import Image

def load_obj(file_path):
    vertices = []
    faces = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('f '):
                face = [list(map(int, v.split('/'))) for v in line.strip().split()[1:]]
                faces.append(face)
    return np.array(vertices), faces

def save_obj(file_path, vertices, faces, uvs):
    with open(file_path, 'w') as file:
        for vertex in vertices:
            file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for uv in uvs:
            file.write(f"vt {uv[0]} {uv[1]}\n")
        for face in faces:
            face_str = ' '.join([f"{v[0]}/{i+1}" for i, v in enumerate(face)])
            file.write(f"f {face_str}\n")

def create_uvs_from_image(vertices, image_path):
    image = Image.open(image_path)
    width, height = image.size
    uvs = []
    for vertex in vertices:
        u = vertex[0] / width
        v = 1 - (vertex[1] / height)
        uvs.append([u, v])
    return np.array(uvs)

def makeUVs(obj_file_path,image_file_path,output_file_path,file):
    # OBJファイルの読み込み
    vertices, faces = load_obj(obj_file_path)

    # UV座標の作成
    uvs = create_uvs_from_image(vertices, image_file_path)

    # 新しいOBJファイルの保存
    save_obj(output_file_path, vertices, faces, uvs)

if __name__ == "__main__":
    print('main.pyを実行してください')