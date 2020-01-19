---
title: TensorFlow Serving入门
tags:
  - tf-serving
mathjax: true
comments: false
copyright: false
date: 2020-01-19 21:19:24
categories: 语言框架
---


### 1. 环境

1. MacOs=10.15.2
2. python=3.7.6
3. tensorflow=1.15.0
4. tensorflow-serving-api=1.15.0
5. docker=19.03.5


### 2. tf-serving介绍

大家习惯使用TensorFlow进行模型的训练、验证和预测，但模型完善之后的生产上线流程就见仁见智了，针对这种情况Google提供了TensorFlow Servering，可以将训练好的模型直接上线并提供服务。

<img src="/posts_res/2020-01-19-TensorFlow-Serving入门/01.webp" />

Tf-Serving的工作流程主要分为以下几个步骤：

1. Source会针对需要进行加载的模型创建一个Loader，Loader中会包含要加载模型的全部信息；
2. Source通知Manager有新的模型需要进行加载；
3. Manager通过版本管理策略（Version Policy）来确定哪些模型需要被下架，哪些模型需要被加载；
4. Manger在确认需要加载的模型符合加载策略，便通知Loader来加载最新的模型；
5. 客户端像服务端请求模型结果时，可以指定模型的版本，也可以使用最新模型的结果；

Tf-Serving客户端和服务端的通信方式有两种(`gRPC`、`RESTfull API`)


### 3. 准备环境和样例

1. 准备 tf-serving 的Docker环境
  ```shell
  docker pull tensorflow/serving
  ```

2. 下载官方示例代码

  ```shell
  mkdir -p /tmp/tfserving
  cd /tmp/tfserving
  git clone https://github.com/tensorflow/serving
  ```


### 4. 启动`RESTfull API`形式的tf-serving

1. 运行 tf-serving
  ```shell
  docker run -p 8501:8501 \
  --mount type=bind,\
  source=/tmp/tfserving/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_cpu,\
  target=/models/half_plus_two \
  -e MODEL_NAME=half_plus_two -t tensorflow/serving &
  ```

  - a. `-p 8501:8501`是将外部端口8501和docker端口8501进行绑定
  - b. `source`指的是外部物理机上模型的路径，`target`指的是docker内部的模型路径；上面的命令将外部的模型路径和docker内部的路径绑定
  - c. `-e`设置docker内部环境变量
  - d. `-t`分配一个伪终端(Allocate a pseudo-TTY)

  - i. 较早的docker版本没有 `--mount` 选项，如遇到提示无此选项，请升级docker版本
  - ii. 如遇到直接`exit 125`的情况，把相对路径换成绝对路径，去掉 `\` 命令换行

2. client端调用tf-serving验证
  ```shell
  # 上面的模型是 y = 0.5 * x + 2
  curl -d '{"instances": [1.0, 2.0, 5.0]}' -X POST http://localhost:8501/v1/models/half_plus_two:predict
  # 输出为：
  # { "predictions": [2.5, 3.0, 4.5] }
  ```


### 5. 启动`gRPC`形式的tf-serving

1. 编译模型
  ```shell
  python /tmp/tfserving/serving/tensorflow_serving/example/mnist_saved_model.py /tmp/tfserving/models/mnist
  ```

  - a. 更多关于提供模型看[Serving a TensorFlow Model](https://www.tensorflow.org/tfx/serving/serving_basic)

2. 运行 tf-serving
  ```shell
  docker run -p 8500:8500 \
  --mount type=bind,source=/tmp/tfserving/models/mnist,target=/models/mnist \
  -e MODEL_NAME=mnist -t tensorflow/serving
  ```

3. client端调用tf-serving验证
  ```shell
  python /tmp/tfserving/serving/tensorflow_serving/example/mnist_client.py --num_tests=1000 --server=127.0.0.1:8500
  # 输出为：
  # Inference error rate: 10.4%
  ```

  - a. 如果直接运行mnist_client.py出现找不到 `tensorflow_serving` 的问题，请手动安装 `pip install tensorflow-serving-api==1.15.0`


### 6. 其他关于 tf-serving 的资料

> 1. [tensorflow/serving](https://github.com/tensorflow/serving)
2. [tensorflow/serving with docker](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/docker.md)
3. [tensorflow serving官网文档](https://www.tensorflow.org/tfx/serving/docker)
4. [TensorFlow Serving-简书 by EddyLiu2017](https://www.jianshu.com/p/afe80b2ed7f0)
5. [一个简单的TensorFlow-Serving例子](https://www.codelast.com/%E5%8E%9F%E5%88%9B-%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84tensorflow-serving%E4%BE%8B%E5%AD%90/)

