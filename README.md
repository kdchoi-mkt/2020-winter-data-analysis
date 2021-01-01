# 2020-winter-data-analysis
20년 겨울에 하는 데이터 분석 및 머신러닝 공부

# 협업 방법 (중요!!!)
터미널에 다음 코드를 로컬 저장소에서 실행시킨다.
```
git init
git remote add origin https://github.com/kdchoi-mkt/2020-winter-data-analysis.git
git pull origin master
```

레포지토리 안의 코드를 수정할 경우에는 다음과 같이 터미널에 코드를 작성하여 새로운 브랜치를 작성한다.
```
git checkout -b <branch_name>
```
그 후, 코드를 수정해서 master 레포지토리에 병합할 경우에는 다음과 같이 코드를 작성하여 pull request를 요청한다.
```
git push origin <branch_name>
```
