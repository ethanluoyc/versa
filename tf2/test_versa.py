import unittest
from data import get_data
from model import Model, InferenceNetwork, OmniGlotEncoder, InferenceBlock
import tensorflow as tf
import inference
import tensorflow_probability as tfp
from utilities import sample_normal, multinoulli_log_density

tasks_per_batch = 16
eval_samples = 5
way = 5
shot = 2
d_theta = 256

def evaluate_task(model, inputs, d_theta, way, samples):
    train_inputs, train_outputs, test_inputs, test_outputs = inputs
    features_train = model.encoder(train_inputs)
    features_test = model.encoder(test_inputs)
    # Infer classification layer from q
    classifier = inference.infer_classifier(
        model.infer_net, features_train, train_outputs, d_theta, way
    )

    # Local reparameterization trick
    # Compute parameters of q distribution over logits
    # (f, num_classes), (num_classes, )
    weight_mean, bias_mean = classifier["weight_mean"], classifier["bias_mean"]
    # (f, num_classes), (num_classes, )
    weight_log_variance, bias_log_variance = (
        classifier["weight_log_variance"],
        classifier["bias_log_variance"],
    )
    # Compute the distribution of the logits
    # given w ~ N(mu, sigma^2)
    # and logits = h.T @ w + b
    # (b, num_classes,)
    logits_mean_test = tf.matmul(features_test, weight_mean) + bias_mean
    #                            (b, f) x (f, num_classes)     (num_classes,)
    # (b, num_classes,)
    logits_log_var_test = tf.math.log(
        tf.matmul(features_test ** 2, tf.exp(weight_log_variance))
        + tf.exp(bias_log_variance)
    )
    # (samples, b, num_classes,)
    logits_sample_test = sample_normal(
        logits_mean_test, logits_log_var_test, samples
    )
    # print(logits_sample_test.shape)
    # (samples, b, num_classes,)
    test_labels_tiled = tf.tile(
        tf.expand_dims(test_outputs, 0), [samples, 1, 1]
    )
    # (samples, b)
    task_log_py = multinoulli_log_density(
        inputs=test_labels_tiled, logits=logits_sample_test
    )
    # (b, num_classes)
    averaged_predictions = tfp.math.reduce_logmeanexp(
        input_tensor=logits_sample_test, axis=0
    )
    # print(averaged_predictions.shape)
    task_accuracy = tf.reduce_mean(
        input_tensor=tf.cast(
            tf.equal(
                tf.argmax(input=test_outputs, axis=-1),
                tf.argmax(input=averaged_predictions, axis=-1),
            ),
            tf.float32,
        )
    )
    # (b,)
    # This is eqn 4
    task_score = tfp.math.reduce_logmeanexp(input_tensor=task_log_py, axis=0)
    # ()
    task_loss = -tf.reduce_mean(input_tensor=task_score, axis=0)
    return [task_loss, task_accuracy]

class DataTest(unittest.TestCase):
    @unittest.skip("Skip")
    def test_load_data(self):
        train_inputs, test_inputs, train_outputs, test_outputs = dataset.get_batch(
            "train", tasks_per_batch, shot, way, eval_samples)
        print(train_inputs.shape, test_inputs.shape, train_outputs.shape, test_outputs.shape)
        self.assertEqual(train_inputs.shape, (tasks_per_batch, way * shot, 28, 28, 1))
        self.assertEqual(test_inputs.shape, (tasks_per_batch, way * eval_samples, 28, 28, 1))
        self.assertEqual(train_outputs.shape, (tasks_per_batch, way * shot, way))
        self.assertEqual(test_outputs.shape, (tasks_per_batch, way * eval_samples, way))

    @unittest.skip("Skip")
    def test_model_output(self):
        dataset = get_data("Omniglot")
        train_inputs, test_inputs, train_outputs, test_outputs = dataset.get_batch(
            "train", tasks_per_batch, shot, way, eval_samples)
        model = Model(d_theta)
        features = model.encoder(train_inputs[0])
        model.infer_net.weight_mean(features)
        # features_train = model.encoder(train_inputs)
        # features_test = model.encoder(test_inputs)

    @unittest.skip("Skip")
    def test_infer_classifier(self):
        train_inputs = tf.zeros((tasks_per_batch, way * shot, 28, 28, 1))
        train_outputs = tf.zeros((tasks_per_batch, way * shot, way))
        # test_inputs = tf.zeros((tasks_per_batch, way * shot, 28, 28, 1))
        # test_outputs = tf.zeros((tasks_per_batch, way * shot, way))
        model = Model(d_theta)
        features = model.encoder(train_inputs[0])
        classifier = inference.infer_classifier(
            model.infer_net, features, train_outputs[0], d_theta, way
        )
        self.assertEqual(classifier['weight_mean'].shape, (d_theta, way))
        self.assertEqual(classifier['weight_log_variance'].shape, (d_theta, way))
        self.assertEqual(classifier['bias_mean'].shape, (way, ))
        self.assertEqual(classifier['bias_log_variance'].shape, (way, ))

    def test_evaluate_tasks(self):
        train_inputs = tf.zeros((tasks_per_batch, way * shot, 28, 28, 1))
        train_outputs = tf.zeros((tasks_per_batch, way * shot, way))
        test_inputs = tf.zeros((tasks_per_batch, way * shot, 28, 28, 1))
        test_outputs = tf.zeros((tasks_per_batch, way * shot, way))
        # inputs = (train_inputs, train_outputs, test_inputs, test_outputs)
        inputs = (train_inputs[0], train_outputs[0], test_inputs[0], test_outputs[0])
        d_theta = 256
        samples = 7
        model = Model(d_theta)
        evaluate_task(model, inputs, d_theta, way, samples)


if __name__ == "__main__":
    unittest.main()