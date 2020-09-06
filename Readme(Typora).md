# Neo4j相关py文件简介

#### 1.库

py2neo  (初次运行请先配置neo4j环境，进行初次网址登陆初次登陆账号：neo4j  密码：neo4j)

pyecharts  (根据neo4j中的节点和关系构建力导向模型)

PyQt5  (构建qt界面，推荐安装5.10.1版本)

```
 pip install pyqt5==5.10.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

 json  (读取json数据库)



#### 2.Batch_upload_neo4j.py（批量上传节点及关系）

​    对于本地all_data(1).json文件进行批量建立节点和关系，于网址http://localhost:7474查看结果，只录入其中“制造”的关系。



#### 3.neo4j_admin.py（模糊查找、精确查找、节点删除）

​    实现对于节点的<u>模糊查找</u>，结果为模糊结果list，再根据模糊查找的结果中选取<u>精确查找</u>的输入，结果输出为字典。

例：{'name': '06系列迷彩服', 'properties': "{'国家': '中国', '生产年限': '2006年', '名称': '06系列迷彩服'}"}

​     实现对于节点的<u>删除</u>。



#### 4.neo4j_echart.py（pyecharts构建html的知识图谱可视化）

​    查询节点并访问其所有父节点和子节点，构建关系转换在同目录下生成echart图谱，推荐输入为国家。



#### 5.qt_html.py（于qt窗口加载本地的html）

  用于在qt窗口加载本地的html。

