# OPT+W
### 미활용 폐교를 이용한 노인 복지 시설 최적 입지 선정

급변하는 인구로 인해 지역별 공공시설 및 서비스 격차가 발생합니다. 이를 보완하기 위해 공공 데이터를 기반으로 변화하는 인구구성비에 맞춰 공공 서비스를 새롭게 분배하고자 하며, 해당 문제를 해결하기 위해서 지역사회의 미활용 폐교를 노인복지시설로 활용하는 아이디어를 제안합니다.  
OPT+W, Optimal welfare 프로젝트는 데이터 분석과 딥러닝을 사용하여 미활용 폐교를 활용한 노인복지시설의 최적 입지 선정을 추천하여 해당 입지의 노인 복지 시설 홈페이지를 제안합니다. 최적의 입지를 선정하기 위해 인구 대비 복지시설이 부족한 지역을 찾고, 위성 사진을 학습한 딥러닝을 사용합니다. 도심에 더 가까운 지역을 점수화하여 공공 서비스가 적절하게 필요하다고 생각되는 지역을 선정하였습니다.


## Data Collection & Analysis Process
- 폐교 현황 수(2019, 지방교육재정알리미) 👉 미활용 폐교 지역
- 주민등록인구현황(2020, 통계청) 👉 지역별 노인 인구 수 & 비율
- 노인복지시설(2019, 보건복지부) 👉 지역별 노인복지시설 개수
- QGIS 지도 위성 데이터 👉 인구 밀집 이미지 분석

### Timeline
- 2020.11 ~ 12 : 프로젝트 완료
- 2022.12 : 프로젝트 리팩토링

```bash
.
├─.idea
│  └─inspectionProfiles
├─dataset
├─data_preprocessing
├─model
│  ├─population_classification.py
│  └─segmentation_classification.py
└─utils
```

## details
class_population = (cls) -> 2개의 이미지 classification
class_segmentation = (cls) -> 4개의 이미지 classification
test data 정확도, precision, recall, f1-score 결과

