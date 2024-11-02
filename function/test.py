import cv2
import numpy as np
import maxflow
import os
from glob import glob

def load_images(folder_path):
    """画像をフォルダから読み込みます。"""
    file_paths = glob(folder_path + '/*.png')
    images = [cv2.imread(file) for file in file_paths]
    return images, file_paths

def calculate_edge_capacity(pixel1, pixel2, delta=1.0):
    """エッジ容量の計算 (式(3)の例として実装)"""
    diff = np.abs(pixel1 - pixel2)
    return delta**2 / (2 * np.linalg.norm(diff) + 1e-5)

def graph_cut(image):
    """グラフカット処理を実行し、画像を二分割します。"""
    h, w, c = image.shape
    graph = maxflow.Graph[float]()
    nodes = graph.add_grid_nodes((h, w))
    
    # ノード間のエッジ容量を設定する
    for i in range(h):
        for j in range(w):
            if j < w - 1:  # 右隣の画素とのエッジ
                capacity = calculate_edge_capacity(image[i, j], image[i, j + 1])
                graph.add_edge(nodes[i, j], nodes[i, j + 1], capacity, capacity)
            if i < h - 1:  # 下隣の画素とのエッジ
                capacity = calculate_edge_capacity(image[i, j], image[i + 1, j])
                graph.add_edge(nodes[i, j], nodes[i + 1, j], capacity, capacity)
    
    # 始点ノードと終点ノードに接続
    for i in range(h):
        for j in range(w):
            graph.add_tedge(nodes[i, j], np.inf if image[i, j, 0] < 128 else 0, 0 if image[i, j, 0] < 128 else np.inf)

    # MinCut計算
    graph.maxflow()
    segments = np.zeros((h, w), np.uint8)
    for i in range(h):
        for j in range(w):
            segments[i, j] = 255 if graph.get_segment(nodes[i, j]) == 0 else 0

    return segments

def process_and_save_images(folder_path, output_folder):
    """フォルダ内の画像に対してグラフカットを適用し、結果を保存します。"""
    images, file_paths = load_images(folder_path)
    os.makedirs(output_folder, exist_ok=True)
    
    for i, (image, file_path) in enumerate(zip(images, file_paths)):
        segmented_image = graph_cut(image)
        output_path = os.path.join(output_folder, f"segmented_{os.path.basename(file_path)}")
        cv2.imwrite(output_path, segmented_image)
        print(f"Processed and saved: {output_path}")

# 実行
input_folder = './output/ss/'  # Regrowで生成した画像が格納されているフォルダ
output_folder = './output/segmented/'
process_and_save_images(input_folder, output_folder)
