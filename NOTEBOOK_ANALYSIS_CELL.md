# 노트북에 추가할 통계 분석 셀

다음 셀을 노트북에 추가하여 통계 분석을 실행할 수 있습니다.

## 셀 추가 위치
통계적 특징 추출(Cell 22) 이후에 추가하세요.

---

## 새 셀: 통계 분석 실행

```python
# 통계 분석 실행
import sys
from pathlib import Path

# 프로젝트 경로 확인
sys.path.insert(0, '/content/gaussian-splatting')

from src.statistical_analysis import run_full_analysis

# records.csv 경로 설정
csv_path = Path("/content/gaussian-splatting/data/ckpts/records.csv")

# 또는 outputs 폴더에 있다면
# csv_path = Path("/content/gaussian-splatting/outputs/records.csv")

if csv_path.exists():
    print(f"📊 통계 분석 시작...")
    print(f"   데이터 파일: {csv_path}")
    
    # 분석 실행 (Google Drive에 자동 저장)
    run_full_analysis(
        csv_path=csv_path,
        output_dir=Path("/content/gaussian-splatting/outputs"),
        save_to_drive_path="/content/drive/MyDrive/3dgs_analysis"  # Google Drive 저장 경로
    )
    
    print("\n✅ 통계 분석 완료!")
    print("   결과 확인:")
    print("   - 로컬: outputs/analysis_report.txt, outputs/figures/")
    print("   - Google Drive: /content/drive/MyDrive/3dgs_analysis/")
else:
    print(f"❌ 파일을 찾을 수 없습니다: {csv_path}")
    print("   Cell 22를 먼저 실행하여 records.csv를 생성해주세요.")
```

---

## 필요한 라이브러리 설치 셀 (분석 전에 실행)

```python
# 통계 분석에 필요한 라이브러리 설치
!pip install -q pygam statsmodels seaborn

print("✅ 라이브러리 설치 완료")
```

