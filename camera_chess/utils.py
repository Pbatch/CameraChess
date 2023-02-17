def get_square(idx):
    x = idx // 8
    y = idx % 8
    return f'{chr(x + 97)}{8 - y}'
