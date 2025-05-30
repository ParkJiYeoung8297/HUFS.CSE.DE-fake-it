import pandas as pd
import subprocess
import sys

# 파라미터 파일 읽기
df = pd.read_excel("params.xlsx")


for idx, row in df.iterrows():
    # row 값을 리스트로 추출 
    params = [str(v) for v in row.values]

    # 첫 5개 파라미터 전체: train_model.py
    train_params = " ".join(params)

    # 첫 4개만: test_model.py
    test_params = " ".join(params[:4])

    print(f"\n==== [{idx+1}/{len(df)}] Running with: {train_params} ====\n")

    print("Training...",flush=True)
    subprocess.run(f"python3 train_model.py {train_params}", shell=True, check=True)

    print("Testing...",flush=True)
    subprocess.run(f"python3 test_model.py {test_params}", shell=True, check=True)