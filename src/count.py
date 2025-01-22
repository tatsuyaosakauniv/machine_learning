# ファイル名を指定
filename = "data/combined_0.1_3000man.dat"

# 行数をカウント（メモリ効率を重視）
line_count = 0
with open(filename, 'r') as file:
    for _ in file:
        line_count += 1

print(f"3000: {line_count}")

filename = "data/combined_0.1_4000man.dat"

# 行数をカウント（メモリ効率を重視）
line_count = 0
with open(filename, 'r') as file:
    for _ in file:
        line_count += 1

print(f"4000: {line_count}")

filename = "data/combined_0.1_5000man.dat"

# 行数をカウント（メモリ効率を重視）
line_count = 0
with open(filename, 'r') as file:
    for _ in file:
        line_count += 1

print(f"5000: {line_count}")