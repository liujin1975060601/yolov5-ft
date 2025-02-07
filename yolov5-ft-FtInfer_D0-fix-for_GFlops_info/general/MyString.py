import os

def replace_path(src_name, last_path, ext):
    # 将源文件名分割为目录、文件名和扩展名
    dirname, filename = os.path.split(src_name)
    base, file_extension = os.path.splitext(filename)

    # 查找并替换最后一个路径部分
    parts = dirname.split(os.path.sep)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = last_path
            break

    # 重新构建目录路径
    new_dirname = os.path.sep.join(parts)

    # 构建新文件名
    new_filename = base + ext

    # 拼接新的文件路径
    dst_name = os.path.join(new_dirname, new_filename)

    return dst_name

def replace_last_path(path,new_folder_name):
    # 将路径拆分成目录和文件名部分
    dirname, basename = os.path.split(path)

    # 将"images"替换为"labels"
    new_path = os.path.join(dirname, new_folder_name)
    return new_path

