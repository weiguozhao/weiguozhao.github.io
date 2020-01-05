---
title: 分布式TensorFlow模式
tags:
  - TensorFlow
mathjax: true
comments: false
copyright: true
date: 2020-01-05 17:41:24
categories: 语言框架
---

### 1. 简介

Tensorflow API提供了[Cluster](https://www.tensorflow.org/api_docs/python/tf/train/ClusterSpec)、
[Server](https://www.tensorflow.org/api_docs/python/tf/distribute/Server?version=stable)以及
[Supervisor](https://www.tensorflow.org/api_docs/python/tf/compat/v1/train/Supervisor)来支持模型的分布式训练。
关于Tensorflow的分布式训练介绍可以参考 [Distributed Tensorflow](https://www.tensorflow.org/guide/distributed_training)。

简单的概括说明如下：
- Tensorflow分布式Cluster由多个Task组成，每个Task对应一个tf.train.Server实例, 作为Cluster的一个单独节点；
- 多个相同作用的Task可以被划分为一个job，例如ps job作为参数服务器只保存Tensorflow model的参数，而worker job则作为计算节点只执行计算密集型的Graph计算。
- Cluster中的Task会相对进行通信，以便进行状态同步、参数更新等操作。

Tensorflow分布式集群的所有节点执行的代码是相同的。分布式任务代码具有固定的模式：

```python
# 第1步：命令行参数解析，获取集群的信息ps_hosts和worker_hosts，以及当前节点的角色信息job_name和task_index

# 第2步：创建当前task结点的Server
cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

# 第3步：如果当前节点是ps，则调用server.join()无休止等待；如果是worker，则执行第4步。
if FLAGS.job_name == "ps":
    server.join()

# 第4步：则构建要训练的模型
# build tensorflow graph model

# 第5步：创建tf.train.Supervisor来管理模型的训练过程
# Create a "supervisor", which oversees the training process.
sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0), logdir="/tmp/train_logs")

# The supervisor takes care of session initialization and restoring from a checkpoint.
sess = sv.prepare_or_wait_for_session(server.target)

# Loop until the supervisor shuts down
while not sv.should_stop()
     # train model
```


### 2. Tensorflow分布式训练代码框架

根据上面说到的Tensorflow分布式训练代码固定模式，如果要编写一个分布式的Tensorlfow代码，其框架如下所示:

```python
import tensorflow as tf

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS


def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":
    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # Build model...
      loss = ...
      global_step = tf.Variable(0)

      train_op = tf.train.AdagradOptimizer(0.01).minimize(
          loss, global_step=global_step)

      saver = tf.train.Saver()
      summary_op = tf.merge_all_summaries()
      init_op = tf.initialize_all_variables()

    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir="/tmp/train_logs",
                             init_op=init_op,
                             summary_op=summary_op,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=600)

    # The supervisor takes care of session initialization and restoring from
    # a checkpoint.
    sess = sv.prepare_or_wait_for_session(server.target)

    # Start queue runners for the input pipelines (if any).
    sv.start_queue_runners(sess)

    # Loop until the supervisor shuts down (or 1000000 steps have completed).
    step = 0
    while not sv.should_stop() and step < 1000000:
      # Run a training step asynchronously.
      # See `tf.train.SyncReplicasOptimizer` for additional details on how to
      # perform *synchronous* training.
      _, step = sess.run([train_op, global_step])


if __name__ == "__main__":
  tf.app.run()
```


对于所有Tensorflow分布式代码，可变的只有两点：
1. 构建tensorflow graph模型代码；
2. 每一步执行训练的代码；

### 3. 分布式MNIST任务

我们通过修改[tensorflow/tensorflow](https://github.com/tensorflow/tensorflow)提供的mnist_softmax.py来构造分布式的MNIST样例来进行验证。
修改后的代码请参考如下：

```python
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

tf.app.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS

def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")

    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

    # Create and start a server for the local task.
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)
    print("Cluster job: %s, task_index: %d, target: %s" % (FLAGS.job_name, FLAGS.task_index, server.target))
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        
        # Assigns ops to the local worker by default.
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)):

            # Build model ...
            mnist = input_data.read_data_sets("data", one_hot=True)
            
            # Create the model
            x = tf.placeholder(tf.float32, [None, 784])
            W = tf.Variable(tf.zeros([784, 10]))
            b = tf.Variable(tf.zeros([10]))
            y = tf.matmul(x, W) + b

            # Define loss and optimizer
            y_ = tf.placeholder(tf.float32, [None, 10])
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

            global_step = tf.Variable(0)

            train_op = tf.train.AdagradOptimizer(0.01).minimize(
                cross_entropy, global_step=global_step)

            # Test trained model
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
            saver = tf.train.Saver()
            summary_op = tf.merge_all_summaries()
            init_op = tf.initialize_all_variables()

        # Create a "Supervisor", which oversees the training process.
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir="/opt/tensor",
                                 init_op=init_op,
                                 summary_op=summary_op,
                                 saver = saver,
                                 global_step=global_step,
                                 save_model_secs=600)

        # The supervisor takes care of session initialization and restoring from
        # a checkpoint.
        sess = sv.prepare_or_wait_for_session(server.target)

        # Start queue runners for the input pipelines (if ang).
        sv.start_queue_runners(sess)

        # Loop until the supervisor shuts down (or 2000 steps have completed).
        step = 0
        while not sv.should_stop() and step < 2000:
            batch_xs, batch_ys = mnist.train.next_batch(100)
            _, step = sess.run([train_op, global_step], feed_dict={x: batch_xs, y_: batch_ys})
            print("Step %d in task %d" % (step, FLAGS.task_index))
        print("done.")
        if FLAGS.task_index != 0:
            print("accuracy: %f" % sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                             y_: mnist.test.labels}))


if __name__ == "__main__":
    tf.app.run()
```

我们同样通过tensorlfow的Docker image来启动一个容器来进行验证。

```
$ docker run -d -v /path/to/your/code:/tensorflow/mnist --name tensorflow tensorflow/tensorflow
```

启动tensorflow之后，启动4个Terminal，然后通过下面命令进入tensorflow容器，切换到/tensorflow/mnist目录下

```bash
$ docker exec -ti tensorflow /bin/bash
$ cd /tensorflow/mnist
```

然后在四个Terminal中分别执行下面一个命令来启动Tensorflow cluster的一个task节点，

```bash
# Start ps 0
python mnist_dist.py --ps_hosts=localhost:2221,localhost:2222 --worker_hosts=localhost:2223,localhost:2224 --job_name=ps --task_index=0

# Start ps 1
python mnist_dist.py --ps_hosts=localhost:2221,localhost:2222 --worker_hosts=localhost:2223,localhost:2224 --job_name=ps --task_index=1

# Start worker 0
python mnist_dist.py --ps_hosts=localhost:2221,localhost:2222 --worker_hosts=localhost:2223,localhost:2224 --job_name=worker --task_index=0

# Start worker 1
python mnist_dist.py --ps_hosts=localhost:2221,localhost:2222 --worker_hosts=localhost:2223,localhost:2224 --job_name=worker --task_index=1
```


------------------------

> [Tensorflow学习笔记4：分布式Tensorflow](https://www.cnblogs.com/lienhua34/p/6005351.html)
