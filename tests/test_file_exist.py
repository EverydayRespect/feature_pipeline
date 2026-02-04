import os

with open("../sampled_video_file_addresses.txt") as f:
    file_list = os.read(f)

    for p in file_list:
        if not os.path.exists(p):
            print(p)