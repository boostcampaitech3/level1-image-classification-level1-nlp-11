import time

start = time.time()  # 시작 시간 저장

## 코드

seconds = int(time.time() - start)
print(f"소요 시간: {seconds // 60}분 {seconds % 60}초")  # 현재시각 - 시작시간 = 실행 시간