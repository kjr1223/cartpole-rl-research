import sqlite3

conn = sqlite3.connect("marking_data.db")
cursor = conn.cursor()

# 전체 몇 개 저장됐는지
cursor.execute("SELECT COUNT(*) FROM markings")
total = cursor.fetchone()[0]
print(f"총 저장된 마킹 수: {total}개")

# 전체 데이터 출력
print("\n[저장된 마킹 데이터]")
print(f"{'ID':>4} | {'step':>5} | {'카트위치':>8} | {'카트속도':>8} | {'막대각도':>8} | {'막대각속도':>8} | 저장시간")
print("-" * 80)

cursor.execute("SELECT * FROM markings")
rows = cursor.fetchall()
for row in rows:
    id_, step, cp, cv, pa, pv, t = row
    print(f"{id_:>4} | {step:>5} | {cp:>8.3f} | {cv:>8.3f} | {pa:>8.3f} | {pv:>8.3f} | {t}")

# 막대각도 기준으로 좋은 마킹 TOP 5
print("\n[막대가 가장 수직에 가까웠던 TOP 5 마킹]")
cursor.execute("""
    SELECT id, step, cart_pos, pole_ang 
    FROM markings 
    ORDER BY ABS(pole_ang) ASC 
    LIMIT 5
""")
top5 = cursor.fetchall()
for row in top5:
    print(f"  ID={row[0]} | step={row[1]} | 카트위치={row[2]:.3f} | 막대각도={row[3]:.3f}")

conn.close()
