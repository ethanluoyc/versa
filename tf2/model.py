import tensorflow as tf

# h = conv2d_pool_block(h, use_batch_norm, dropout_keep_prob, 'same', 'fe_block_2')
# h = conv2d_pool_block(h, use_batch_norm, dropout_keep_prob, 'same', 'fe_block_3')
# h = conv2d_pool_block(h, use_batch_norm, dropout_keep_prob, 'same', 'fe_block_4')


class Conv2DPoolBlock(tf.Module):
    def __init__(self, dropout_keep_prob, use_batch_norm=True, pool_padding="same", name=None):
        super().__init__(name=name)
        self._dropout_keep_prob = dropout_keep_prob
        self._use_batch_norm = use_batch_norm
        self.conv = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[3, 3],
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="glorot_normal",
        )
        if use_batch_norm:
            self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-5)
        self.dropout = tf.keras.layers.Dropout(self._dropout_keep_prob)
        self.pool = tf.keras.layers.MaxPool2D(
            pool_size=[2, 2], strides=2, padding=pool_padding
        )

    def __call__(self, h, training):
        h = self.conv(h)
        if self._use_batch_norm:
            h = self.bn(h, training)
        # h = tf.nn.dropout(x=h, keep_prob=self._dropout_keep_prob)
        h = self.dropout(tf.nn.relu(h), training=training)
        h = self.pool(h)
        return h


class DenseBlock(tf.Module):
    def __init__(self, output_size, use_batch_norm, dropout_keep_prob, name=None):
        self.use_batch_norm = use_batch_norm
        self.dropout_keep_prob = dropout_keep_prob
        self.dense = tf.keras.layers.Dense(
            output_size,
            use_bias=False,
            kernel_initializer="glorot_normal",
        )
        if use_batch_norm:
            self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-5)
        self.dropout = tf.keras.layers.Dropout(dropout_keep_prob)

    def __call__(self, x, training):
        h = self.dense(x)
        if self.use_batch_norm:
            h = self.bn(h, training)
        return self.dropout(tf.nn.relu(h), training=training)


class OmniGlotEncoder(tf.Module):
    def __init__(
        self, output_size, dropout_keep_prob, use_batch_norm=True, name=None
    ):
        super().__init__(name=name)
        self.conv1 = Conv2DPoolBlock(dropout_keep_prob, use_batch_norm)
        self.conv2 = Conv2DPoolBlock(dropout_keep_prob, use_batch_norm)
        self.conv3 = Conv2DPoolBlock(dropout_keep_prob, use_batch_norm)
        self.conv4 = Conv2DPoolBlock(dropout_keep_prob, use_batch_norm)
        self.flatten = tf.keras.layers.Flatten()
        # self.dense = DenseBlock(output_size, use_batch_norm, dropout_keep_prob)

    def __call__(self, inputs, training):
        x = self.conv1(inputs, training)
        x = self.conv2(x, training)
        x = self.conv3(x, training)
        x = self.conv4(x, training)
        return self.flatten(x)
        # return self.dense(self.flatten(x), training)


class SmallEncoder(tf.Module):
    def __init__(self, output_size, dropout_keep_prob, use_batch_norm=False, name=None):
        super().__init__(name=name)
        self.layers = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=[3, 3],
                    strides=(1, 1),
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(output_size),
            ]
        )

    def __call__(self, inputs, training):
        return self.layers(inputs)


class InferenceBlock(tf.Module):
    def __init__(self, d_theta, output_units, name=None):
        super().__init__(name=name)
        self.layers = [
            tf.keras.layers.Dense(d_theta, activation=tf.nn.elu, use_bias=True),
            tf.keras.layers.Dense(d_theta, activation=tf.nn.elu, use_bias=True),
            tf.keras.layers.Dense(output_units, use_bias=True),
        ]

    def __call__(self, inputs, training=True):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class InferenceNetwork(tf.Module):
    def __init__(self, d_theta, name=None):
        super().__init__(name=name)
        self.weight_mean = InferenceBlock(d_theta, d_theta, name="weight_mean")
        self.weight_log_variance = InferenceBlock(
            d_theta, d_theta, name="weight_log_variance"
        )
        self.bias_mean = InferenceBlock(d_theta, 1, name="bias_mean")
        self.bias_log_variance = InferenceBlock(d_theta, 1, name="bias_log_variance")


class Model(tf.Module):
    def __init__(self, d_theta, use_batch_norm=True, dropout_keep_prob=0.1, name=None):
        super().__init__(name=name)
        self.encoder = OmniGlotEncoder(d_theta, dropout_keep_prob, use_batch_norm)
        # self.encoder = SmallEncoder(d_theta, dropout_keep_prob, use_batch_norm)
        self.infer_net = InferenceNetwork(d_theta)

    # def __call__(self, train_inputs, train_outputs, test_inputs, training=True):
    #     train_inputs, train_outputs, test_inputs, test_outputs = inputs
    #     features_train = model.encoder(train_inputs, training=training)
    #     features_test = model.encoder(test_inputs, training=training)
    #     # Infer classification layer from q
    #     classifier = infer_classifier(model.infer_net, features_train, train_outputs, way)

    #     # Local reparameterization trick
    #     # Compute parameters of q distribution over logits
    #     # (f, num_classes), (num_classes, )
    #     weight_mean, bias_mean = classifier["weight_mean"], classifier["bias_mean"]
    #     # (f, num_classes), (num_classes, )
    #     weight_log_variance, bias_log_variance = (
    #         classifier["weight_log_variance"],
    #         classifier["bias_log_variance"],
    #     )
