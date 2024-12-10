import matplotlib.font_manager as fm

# 使用可能なフォント一覧を取得
available_fonts = fm.findSystemFonts()

font_names = []
print("Available Fonts:")
for font_path in available_fonts:
    try:
        font_name = fm.FontProperties(fname=font_path).get_name()
        font_names.append(font_name)
        print(f"{font_path} -> {font_name}")
    except RuntimeError as e:
        print(f"Error loading font {font_path}: {e}")

print("\nAvailable Font Names:")
for font_name in font_names:
    print(font_name)
