import json

def read_obj_with_labels(file_path, labels):
    vertices = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

    vertex_lines = [line for line in lines if line.startswith('v ')]
    for line in vertex_lines:
        parts = line.split()
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        vertices.append((x, y, z))

    colors = generate_colors(labels)

    return lines, vertices, colors

def generate_colors(labels):
    # 生成颜色的简单方法，确保每个标签有不同的颜色
    import random
    random.seed(0)  # 确保颜色生成的一致性
    label_color_map = {}
    colors = []
    for label in labels:
        if label not in label_color_map:
            label_color_map[label] = (random.random(), random.random(), random.random())
        colors.append(label_color_map[label])
    return colors

def write_obj_with_colors(file_path, lines, vertices, colors):
    color_lines = []
    vertex_index = 0
    for line in lines:
        if line.startswith('v '):
            x, y, z = vertices[vertex_index]
            r, g, b = colors[vertex_index]
            color_line = f"v {x} {y} {z} {r} {g} {b}\n"
            color_lines.append(color_line)
            vertex_index += 1
        else:
            color_lines.append(line)
    
    with open(file_path, 'w') as f:
        f.writelines(color_lines)

# 读取 JSON 文件
with open('/home/repos/ToothGroupNetwork/new.json', 'r') as f:
    data = json.load(f)

labels = data['labels']

# 读取 OBJ 文件并提取顶点和颜色
lines, vertices, colors = read_obj_with_labels('/home/repos/ToothGroupNetwork/new.obj', labels)

# 写入带有颜色的 OBJ 文件
write_obj_with_colors('/home/repos/ToothGroupNetwork/2_colored.obj', lines, vertices, colors)
