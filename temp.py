import os

def check_file_sizes(base_dir):
    frame_dirs = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    
    sizes = []
    dir_size_dict = {}
    for dir in frame_dirs:
        file_path = os.path.join(base_dir, dir, dir + '.msgpack')
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            sizes.append(size)
            dir_size_dict[dir] = size
        else:
            print(f"File {file_path} not found.")

    # 파일 크기가 모두 동일한지 확인합니다.
    if len(set(sizes)) == 1:
        print("All file sizes are identical.")
    else:
        print("File sizes are not identical. Here are the sizes of all files:")
        for dir, size in dir_size_dict.items():
            print(f"Directory: {dir}, File Size: {size}")

# 사용 예시:
check_file_sizes('/data/jychoi/results/train_f_t_t10_10000_2000/models')
