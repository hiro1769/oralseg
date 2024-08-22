import numpy as np
from sklearn.cluster import DBSCAN
import json
import os
import shutil

def load_obj(file_path):
    vertices = []
    faces = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.split()
                faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
    return np.array(vertices), faces

def save_obj(file_path, vertices, faces):
    with open(file_path, 'w') as file:
        for vertex in vertices:
            file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
        for face in faces:
            file.write(f"f {face[0]} {face[1]} {face[2]}\n")

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, separators=(', ', ': '))

def filter_outliers(vertices, faces, labels, instances, eps=1.3, min_samples=40):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels_db = db.fit_predict(vertices)
    filtered_indices = np.where(labels_db != -1)[0]
    filtered_vertices = vertices[filtered_indices]
    filtered_labels = [labels[i] for i in filtered_indices]
    filtered_instances = [instances[i] for i in filtered_indices]

    # 更新面信息
    final_faces = []
    index_map = {old_idx: new_idx + 1 for new_idx, old_idx in enumerate(filtered_indices)}
    for face in faces:
        if all(idx - 1 in index_map for idx in face):
            final_faces.append([index_map[idx - 1] for idx in face])

    return filtered_vertices, final_faces, filtered_labels, filtered_instances

def process_files(obj_file_path, json_file_path, output_obj_dir, output_json_dir, eps=1.3, min_samples=40):
    vertices, faces = load_obj(obj_file_path)
    json_data = load_json(json_file_path)
    labels = json_data['labels']
    instances = json_data['instances']

    # 剔除离群点
    filtered_vertices, filtered_faces, filtered_labels, filtered_instances = filter_outliers(
        vertices, faces, labels, instances, eps, min_samples
    )

    # 创建输出目录结构
    rel_path = os.path.relpath(obj_file_path, obj_dir)
    new_obj_file_path = os.path.join(output_obj_dir, rel_path)
    os.makedirs(os.path.dirname(new_obj_file_path), exist_ok=True)

    new_json_file_path = os.path.join(output_json_dir, rel_path).replace('.obj', '.json')
    os.makedirs(os.path.dirname(new_json_file_path), exist_ok=True)

    # 保存新的 OBJ 和 JSON 文件
    save_obj(new_obj_file_path, filtered_vertices, filtered_faces)
    json_data['labels'] = filtered_labels
    json_data['instances'] = filtered_instances
    save_json(new_json_file_path, json_data)

# 设置输入输出目录
obj_dir = '/home/repos/ToothGroupNetwork/data/3D_scans_per_patient_obj_files' 
json_dir = '/home/repos/ToothGroupNetwork/data/ground-truth_labels_instances' 
output_obj_dir = '/path/to/your/output/obj/directory' 
output_json_dir = '/path/to/your/output/json/directory'

# 遍历所有文件并处理
for patient_id in os.listdir(obj_dir):
    patient_obj_dir = os.path.join(obj_dir, patient_id)
    patient_json_dir = os.path.join(json_dir, patient_id)

    for filename in os.listdir(patient_obj_dir):
        if filename.endswith('.obj'):
            obj_file_path = os.path.join(patient_obj_dir, filename)
            json_file_path = os.path.join(patient_json_dir, filename.replace('.obj', '.json'))

            if os.path.exists(json_file_path):
                print(f"Processing {obj_file_path}...")
                process_files(obj_file_path, json_file_path, output_obj_dir, output_json_dir)