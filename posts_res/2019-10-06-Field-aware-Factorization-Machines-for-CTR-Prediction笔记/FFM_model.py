import os

import numpy as np
import tensorflow as tf

input_x_size = 20
field_size = 2
vector_dimension = 3  # 隐因子的维度
total_plan_train_steps = 1000
# 使用SGD，每一个样本进行依次梯度下降，更新参数
batch_size = 10
all_data_size = 100
lr = 0.01

MODEL_SAVE_PATH = "TFModel"
MODEL_NAME = "FFM"
MODEL_PATH = os.path.join(MODEL_SAVE_PATH, MODEL_NAME)


def batcher(X_, y_=None, batch_size=-1):
    n_samples = X_.shape[0]

    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
        raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X_[i:upper_bound]
        ret_y = None
        if y_ is not None:
            ret_y = y_[i:i + batch_size]
            yield (ret_x, ret_y)


def createTwoDimensionWeight(input_x_size, field_size, vector_dimension):
    weights = tf.truncated_normal([input_x_size, field_size, vector_dimension])
    tf_weights = tf.Variable(weights)
    return tf_weights


def createOneDimensionWeight(input_x_size):
    weights = tf.truncated_normal([input_x_size])
    tf_weights = tf.Variable(weights)
    return tf_weights


def createZeroDimensionWeight():
    weights = tf.truncated_normal([1])
    tf_weights = tf.Variable(weights)
    return tf_weights


def inference(input_x, input_x_field, zeroWeights, oneDimWeights, thirdWeight):
    """计算回归模型输出的值"""
    # 线性乘积结果
    secondValue = tf.reduce_sum(tf.multiply(oneDimWeights, input_x, name='secondValue'))
    # bias + 线性乘积结果
    firstTwoValue = tf.add(zeroWeights, secondValue, name="firstTwoValue")

    thirdValue = tf.Variable(0.0, dtype=tf.float32)
    input_shape = input_x_size

    for i in range(input_shape):
        featureIndex1 = i
        fieldIndex1 = int(input_x_field[i])
        for j in range(i + 1, input_shape):
            featureIndex2 = j
            fieldIndex2 = int(input_x_field[j])
            vectorLeft = tf.convert_to_tensor([[featureIndex1, fieldIndex2, i] for i in range(vector_dimension)])
            weightLeft = tf.gather_nd(thirdWeight, vectorLeft)
            weightLeftAfterCut = tf.squeeze(weightLeft)

            vectorRight = tf.convert_to_tensor([[featureIndex2, fieldIndex1, i] for i in range(vector_dimension)])
            weightRight = tf.gather_nd(thirdWeight, vectorRight)
            weightRightAfterCut = tf.squeeze(weightRight)

            tempValue = tf.reduce_sum(tf.multiply(weightLeftAfterCut, weightRightAfterCut))

            indices2 = [i]
            indices3 = [j]

            xi = tf.squeeze(tf.gather_nd(input_x, indices2))
            xj = tf.squeeze(tf.gather_nd(input_x, indices3))

            product = tf.reduce_sum(tf.multiply(xi, xj))

            secondItemVal = tf.multiply(tempValue, product)

            tf.assign(thirdValue, tf.add(thirdValue, secondItemVal))

    return tf.add(firstTwoValue, thirdValue)


def gen_data():
    labels = [-1, 1]
    # 随机生成 label 序列
    y = [np.random.choice(labels, 1)[0] for _ in range(all_data_size)]
    # 20 维特征 [0, 0, 0, ..., 1, 1, 1] 10个0，10个1
    x_field = [i // 10 for i in range(input_x_size)]
    # 生成 1000 x 20的训练数据
    x = np.random.randint(0, 2, size=(all_data_size, input_x_size))
    return np.asarray(x), np.asarray(y), np.asarray(x_field)


if __name__ == '__main__':
    global_step = tf.Variable(0, trainable=False)

    # 生成训练数据
    trainx, trainy, trainx_field = gen_data()
    print(trainx.shape, len(trainy))

    # 开始构建图
    input_x = tf.placeholder(tf.float32, [None, input_x_size])
    input_y = tf.placeholder(tf.float32, [None, ])

    # 正则项权重的系数
    lambda_w = tf.constant(0.001, name='lambda_w')
    lambda_v = tf.constant(0.001, name='lambda_v')

    # bias/ w_0
    zeroWeights = createZeroDimensionWeight()
    tf.summary.histogram('bias', zeroWeights)

    # weight / w_i
    oneDimWeights = createOneDimensionWeight(input_x_size)
    tf.summary.histogram('oneDimWeights', oneDimWeights)

    # field_weight / v_i
    thirdWeight = createTwoDimensionWeight(input_x_size,  # 创建二次项的权重变量
                                           field_size,
                                           vector_dimension)  # n * f * k
    tf.summary.histogram('oneDimWeights', thirdWeight)

    # 模型推断
    y_ = inference(input_x, trainx_field, zeroWeights, oneDimWeights, thirdWeight)
    # 计算正则
    l2_norm = tf.reduce_sum(
        tf.add(
            tf.multiply(lambda_w, tf.pow(oneDimWeights, 2)),
            tf.reduce_sum(tf.multiply(lambda_v, tf.pow(thirdWeight, 2)), axis=[1, 2])
        )
    )
    # loss项
    loss = tf.reduce_mean(tf.log(1 + tf.exp(input_y * y_)) + l2_norm)
    tf.summary.scalar('loss', loss)
    # 优化器
    train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)

    # 定义saver
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    t = 0
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(MODEL_PATH, sess.graph)
        print("logdir=", MODEL_PATH)
        sess.run(tf.global_variables_initializer())
        for i in range(total_plan_train_steps):
            perm = np.random.permutation(trainx.shape[0])
            for input_x_batch, input_y_batch in batcher(trainx[perm], trainy[perm], batch_size):
                predict_loss, _, steps = sess.run([loss, train_step, global_step],
                                                  feed_dict={input_x: input_x_batch, input_y: input_y_batch})

                print("After {step} training step(s), loss on training batch is {predict_loss}"
                      .format(step=steps, predict_loss=predict_loss))

                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=steps)

                result = sess.run(merged, feed_dict={input_x: input_x_batch, input_y: input_y_batch})
                writer.add_summary(result, t)
                t += 1
        writer.close()
        #
