import sys
from PIL import Image

def show_image(path, width=100):
    """PNG 등 이미지를 터미널에 트루컬러 유니코드 반블록(▀)으로 출력"""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    # 터미널 글자는 세로가 가로보다 길어서 세로 방향을 절반 정도로 줄여 비율 보정
    new_h = int(h * (width / w) * 0.5)
    img = img.resize((width, new_h * 2))
    px = img.load()

    out = []
    for y in range(0, new_h * 2, 2):
        line = []
        for x in range(width):
            r1, g1, b1 = px[x, y]
            r2, g2, b2 = px[x, y + 1]
            # 윗칸=글자색(▀), 아랫칸=배경색으로 세로 2픽셀을 한 글자에 압축
            line.append(f"\x1b[38;2;{r1};{g1};{b1}m\x1b[48;2;{r2};{g2};{b2}m▀")
        out.append("".join(line) + "\x1b[0m")
    print("\n".join(out))

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "root_locus_pd.png"
    width = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    show_image(path, width)
