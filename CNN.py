'''
This script applies cnn model for sleep stage annotation
'''

import tensorflow as tf

#hyperparameters


#####################
# step 1: read input data
#####################


######################
# step 2: define cnn architecture
######################
def cnn_model_fn(features, labels, mode, parameters):
    '''model function for cnn'''
    # input layer
    # shape:
    input_layer = features

    # conv1
    # input tensor shape:
    # output tensor shape
    # try other arguments if possible, e.g., kernel_initializer, strides
    conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=32,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)

    # pool1
    # input tensor shape:
    # output tensor shape:
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # conv2
    #input tensor shape:
    #output tensor shape:
    conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=64,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)

    # pool2
    #
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # conv3
    conv3 = tf.layers.conv2d(
          inputs=pool2,
          filters=64,
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)

    # pool3
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # local1
    pool3_flat = tf.reshape(pool3, [-1, 7 * 7 * 64])
    local1 = tf.layers.dense(inputs=pool3_flat, units=1024, activation=tf.nn.relu)

    # logits layer
    logits = tf.layers.dense(inputs=local1, units=10)

    predictions = {
            "classes": tf.argmax(input=logits, axis=1)
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }
    if mode == tf.estimators.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    # calculate loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimators.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
        train_op = optimizer.minimize(loss=loss, global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # add evaluation metrics
    eval_metric_ops = {"accuracy":tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimators.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():
    #load the data

    train_data = 
    train_labels = 

    eval_data = 
    eval_labels =

    # create the estimator
    eeg_classifer = tf.estimator.Estimator(mode_fn=cnn_model_fn, model_dir="")

    # set up logging info


    # train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x = {"x":train_data},
            y = train_labels,
            batch_size = 100,
            num_epochs = None,
            shuffle = True
            )

    eeg_classifer.train(
            input_fn = train_input_fn,
            steps = 20000,
            hooks =[?]
            )

    # evaluate the model and print the results
    eval_input_fn = tf.estimators.inputs.numpy_input_fn(
            x = {'x': eval_data},
            y = eval_labels,
            num_epochs = 1,
            shuffle = False
            )
    eval_results = eeg_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ = "__main__":
    tf.app.run()
