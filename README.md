# behavior2text

On Keyword Extraction for User Intention Description based on Smartphone Context Logs

## Installing

`pip install behavior2text`

## Running

### Apis

url domain:<http://udiclab.cs.nchu.edu.tw>

1. Generate Sequence with Accessibility logs：_`/behavior2text`_
  
  - example code:
  ```
  秉宏交給你了
  秉宏交給你了
  秉宏交給你了
  秉宏交給你了
  import requests
  requests.post(秉宏交給你了)
  秉宏交給你了
  秉宏交給你了
  秉宏交給你了
  秉宏交給你了
  ```

  - result:
  ```
  在查詢臺東民宿的資訊
  ```

### Commands

1. `python3 manage.py experiment --topNMax=<(optional) at least 2> --clusterTopnMax=<(optional) at least 2>`:
    * show NDCG of sequences when using each kind of methods: tfidf, kcem, kcemCluster, hybrid, contextNetwork, pagerank
    * (Optional Usage): `python3 experiment.py`
2. `python3 manage.py sentence --method=<> --debug=<True/False>`:
    * show sequences with those accessibility using provided method below
    * method:tfidf, kcem, kcemCluster, hybrid, contextNetwork, pagerank
3. `python3 __init__.py <method>`:
    * tfidf
    * kcem
    * kcemCluster
    * hybrid
    * contextNetwork
    * pagerank

### Settings

1. `settings.py`裏面需要新增`behavior2text`這個app：

  - add this:

    ```
    INSTALLED_APPS=[
    ...
    ...
    ...
    'behavior2text',
    ]
    ```

2. `urls.py`需要新增下列代碼 把所有search開頭的request都導向到`behavior2text`這個app：

  - add this:

    ```
    import behavior2text.urls
    urlpatterns += [
        url(r'^behavior2text/', include(behavior2text.urls))
    ]
    ```

3. `python manage.py runserver`：即可進入頁面 `127.0.0.1:8000/behavior2text` 測試 `behavior2text` 是否安裝成功。

## Deployment

`behavior2text` is a django-app, so depends on django project.

`behavior2text` 是一般的django插件，所以必須依存於django專案

## Built With

- numpy
- scipy
- requests
- pyprind
- udicOpenData

## Contributors

- **張泰瑋** [david](https://github.com/david30907d)

## License

This package use `GPL3.0` License.