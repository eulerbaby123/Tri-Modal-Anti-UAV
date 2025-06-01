# import os

# def find_missing_images(label_dir, image_dir, image_exts=None):
#     """
#     遍历 label_dir 下的所有 .txt 文件，检查 image_dir 中是否有同名图片。
#     返回缺失图片的 label 文件名列表。
#     """
#     if image_exts is None:
#         image_exts = ['.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff']
    
#     # 获取所有 label 的基名（不含扩展）
#     label_files = [f for f in os.listdir(label_dir) if f.lower().endswith('.txt')]
#     label_bases = {os.path.splitext(f)[0] for f in label_files}
    
#     # 获取所有 image 的基名到文件名的映射
#     image_files = [f for f in os.listdir(image_dir) 
#                    if os.path.splitext(f)[1].lower() in image_exts]
#     image_map = {os.path.splitext(f)[0]: f for f in image_files}
    
#     # 查找哪些 label 对应不到 image
#     missing = [base for base in sorted(label_bases) if base not in image_map]
#     return missing

# if __name__ == '__main__':
#     label_dir = r'D:\FLIR\labels'
#     image_dir = r'D:\FLIR\images'
#     exts = ['.jpg', '.png']  # 你可根据需要增删扩展名
    
#     if not os.path.isdir(label_dir):
#         print(f"错误：标注目录不存在：{label_dir}")
#         exit(1)
#     if not os.path.isdir(image_dir):
#         print(f"错误：图片目录不存在：{image_dir}")
#         exit(1)
    
#     missing = find_missing_images(label_dir, image_dir, exts)
#     if missing:
#         print("以下标注文件缺少对应的图片：")
#         for base in missing:
#             print(f" - {base}.txt")
#     else:
#         print("所有标注文件都有对应的图片。")
# import os

# def find_missing_events(event_list_file):
#     """
#     逐行读取 event_list_file 中的路径，检查文件是否存在。
#     返回一个列表，元素为 (行号, 路径)。
#     """
#     missing = []
#     with open(event_list_file, 'r', encoding='utf-8') as f:
#         for i, line in enumerate(f, 1):
#             path = line.strip()
#             if not path:
#                 # 如果行为空，跳过
#                 continue
#             if not os.path.exists(path):
#                 missing.append((i, path))
#     return missing

# if __name__ == '__main__':
#     event_list_file = r'D:\FLIR\val_event.txt'
#     if not os.path.isfile(event_list_file):
#         print(f"错误：找不到文件 {event_list_file}")
#         exit(1)

#     missing = find_missing_events(event_list_file)
#     if missing:
#         print("以下行的 event 文件缺失：")
#         for line_no, path in missing:
#             print(f"  第 {line_no} 行：{path}")
#     else:
#         print("✔ 所有 event 文件都存在。")
import os

def find_missing_event_images(event_list_file, image_dir, image_exts=None):
    """
    逐行读取 event_list_file 中的路径或文件名，提取基名，
    检查 image_dir 中是否有同名图片。返回缺失的 (行号, 原始行内容) 列表。
    """
    if image_exts is None:
        image_exts = ['.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff']
    
    missing = []
    with open(event_list_file, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            raw = line.strip()
            if not raw:
                continue  # 跳过空行
            
            # 提取文件基名，例如：
            # 如果 raw 是 D:\FLIR\events\E0001.dat 或 E0001.dat，都能取到 "E0001"
            base = os.path.splitext(os.path.basename(raw))[0]
            
            # 检查 images 目录下是否有 base + 任意扩展名
            found = False
            for ext in image_exts:
                img_path = os.path.join(image_dir, base + ext)
                if os.path.isfile(img_path):
                    found = True
                    break
            
            if not found:
                missing.append((lineno, raw))
    return missing

if __name__ == '__main__':
    event_list_file = r'D:\FLIR\val_rgb.txt'
    image_dir        = r'D:\FLIR\images'
    exts             = ['.jpg', '.png']  # 根据需要增删
    
    if not os.path.isfile(event_list_file):
        print(f"错误：找不到事件列表文件：{event_list_file}")
        exit(1)
    if not os.path.isdir(image_dir):
        print(f"错误：找不到图片目录：{image_dir}")
        exit(1)

    missing = find_missing_event_images(event_list_file, image_dir, exts)
    if missing:
        print("以下行的事件在 images 中没有对应图片：")
        for lineno, raw in missing:
            print(f"  第 {lineno} 行：{raw}")
    else:
        print("✔ 所有事件行在 images 目录中都有对应图片。")
