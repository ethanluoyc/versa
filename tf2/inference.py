import tensorflow as tf


def infer_classifier(infer_net, features, labels, num_classes):
    """
    Infer a linear classifier by concatenating vectors for each class.
    :param features: tensor (tasks_per_batch x num_features) feature matrix
    :param labels:  tensor (tasks_per_batch x num_classes) one-hot label matrix
    :param num_classes: Integer number of classes per task.
    :return: Dictionary containing output classifier layer (including means and
    :        log variances for weights and biases).
    """

    classifier = {}
    class_weight_means = []
    class_weight_logvars = []
    class_bias_means = []
    class_bias_logvars = []
    for c in range(num_classes):
        # labels (batch, num_classes)
        # (batch, num_classes) of bools
        class_mask = tf.equal(tf.argmax(input=labels, axis=1), c)
        # (batch where cls == c, f) of floats
        class_features = tf.boolean_mask(tensor=features, mask=class_mask)

        # Pool across dimensions
        # (1, f)
        nu = tf.expand_dims(tf.reduce_mean(input_tensor=class_features, axis=0), axis=0)
        # (1, num_classes)
        class_weight_means.append(infer_net.weight_mean(nu))
        # (1, num_classes)
        class_weight_logvars.append(infer_net.weight_log_variance(nu))
        # (1, 1)
        class_bias_means.append(infer_net.bias_mean(nu))
        # (1, 1)
        class_bias_logvars.append(infer_net.bias_log_variance(nu))

    # (f, num_classes)
    classifier["weight_mean"] = tf.transpose(a=tf.concat(class_weight_means, axis=0))
    # (num_classes, )
    classifier["bias_mean"] = tf.reshape(
        tf.concat(class_bias_means, axis=1),
        [
            num_classes,
        ],
    )
    # (f, num_classes)
    classifier["weight_log_variance"] = tf.transpose(
        a=tf.concat(class_weight_logvars, axis=0)
    )
    # (num_classes,)
    classifier["bias_log_variance"] = tf.reshape(
        tf.concat(class_bias_logvars, axis=1),
        [
            num_classes,
        ],
    )

    return classifier
