import os

# 要处理的 txt 文件路径列表
txt_files = [
    r"C:\Users\xwj\Desktop\autodl\Tri-Modal-Anti-UAV1\train_event.txt",
    r"C:\Users\xwj\Desktop\autodl\Tri-Modal-Anti-UAV1\train_ir.txt",
    r"C:\Users\xwj\Desktop\autodl\Tri-Modal-Anti-UAV1\train_rgb.txt",
    r"C:\Users\xwj\Desktop\autodl\Tri-Modal-Anti-UAV1\val_event.txt",
    r"C:\Users\xwj\Desktop\autodl\Tri-Modal-Anti-UAV1\val_ir.txt",
    r"C:\Users\xwj\Desktop\autodl\Tri-Modal-Anti-UAV1\val_rgb.txt"
]

# 替换前缀（Linux 路径）
old_prefix = "/root/autodl-tmp/multispectral-object-detection-main/dataset/uav_online/images/"

# 替换后缀（Windows 本地路径）
new_prefix = r"C:\Users\xwj\Desktop\autodl\Tri-Modal-Anti-UAV1\images"

# 遍历每个 txt 文件并修改内容
for file_path in txt_files:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        line = line.strip()
        if line.startswith(old_prefix):
            filename = os.path.basename(line)
            new_line = os.path.join(new_prefix, filename)
            new_lines.append(new_line)
        else:
            new_lines.append(line)

    # 回写覆盖原文件
    with open(file_path, "w", encoding="utf-8") as f:
        for line in new_lines:
            f.write(line + "\n")

print("✅ 所有路径已成功替换为本地地址。")
