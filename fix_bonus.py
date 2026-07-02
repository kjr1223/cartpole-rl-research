# swingup_marking_sac.py의 보너스 조건 수정
with open("swingup_marking_sac.py", "r") as f:
    code = f.read()

# 기존 보너스 코드
old = """            # 마킹 보너스 추가 (마킹 SAC만)
            if use_marking:
                r += get_bonus(obs, markings)"""

# 수정된 보너스 코드
# swingup 모드 + 전환 기준(±20도) 근처일 때만 보너스
new = """            # 마킹 보너스 추가 (마킹 SAC만)
            # 조건: swingup 모드 + 각도 ±40도 이내일 때만
            # 전환 직전 순간에만 보너스를 줘야 올바르게 학습
            if use_marking and env.mode == "swingup" and abs(obs[2]) < np.radians(40):
                r += get_bonus(obs, markings)"""

code = code.replace(old, new)
with open("swingup_marking_sac.py", "w") as f:
    f.write(code)
print("수정 완료!")
