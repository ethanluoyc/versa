"""
TODOs:

1. Match architectures
2. Match training parameters

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import tensorflow as tf
import argparse

from inference import infer_classifier
from data import get_data
from model import Model
import tensorflow_probability as tfp

"""
parse_command_line: command line parser
"""


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        "-d",
        choices=["Omniglot", "miniImageNet"],
        default="Omniglot",
        help="Dataset to use",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test", "train_test"],
        default="train_test",
        help="Whether to run traing only, testing only, or both training and testing.",
    )
    parser.add_argument(
        "--d_theta", type=int, default=256, help="Size of the feature extractor output."
    )
    parser.add_argument(
        "--shot", type=int, default=5, help="Number of training examples."
    )
    parser.add_argument("--way", type=int, default=5, help="Number of classes.")
    parser.add_argument(
        "--test_shot",
        type=int,
        default=None,
        help="Shot to be used at evaluation time. If not specified 'shot' will be used.",
    )
    parser.add_argument(
        "--test_way",
        type=int,
        default=None,
        help="Way to be used at evaluation time. If not specified 'way' will be used.",
    )
    parser.add_argument(
        "--tasks_per_batch", type=int, default=16, help="Number of tasks per batch."
    )
    parser.add_argument(
        "--samples", type=int, default=10, help="Number of samples from q."
    )
    parser.add_argument(
        "--learning_rate", "-lr", type=float, default=1e-4, help="Learning rate."
    )
    parser.add_argument(
        "--iterations", type=int, default=80000, help="Number of training iterations."
    )
    parser.add_argument(
        "--checkpoint_dir",
        "-c",
        default="./checkpoint",
        help="Directory to save trained models.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout keep probability."
    )
    parser.add_argument(
        "--test_model_path", "-m", default=None, help="Model to load and test."
    )
    parser.add_argument(
        "--print_freq",
        type=int,
        default=200,
        help="Frequency of summary results (in iterations).",
    )
    args = parser.parse_args()

    # adjust test_shot and test_way if necessary
    if args.test_shot is None:
        args.test_shot = args.shot
    if args.test_way is None:
        args.test_way = args.way

    return args


def evaluate_task(model, inputs, way, samples, training):
    train_inputs, train_outputs, test_inputs, test_outputs = inputs
    features_train = model.encoder(train_inputs, training=training)
    features_test = model.encoder(test_inputs, training=training)
    # Infer classification layer from q
    classifier = infer_classifier(model.infer_net, features_train, train_outputs, way)

    # Local reparameterization trick
    # Compute parameters of q distribution over logits
    # (f, num_classes), (num_classes, )
    weight_mean, bias_mean = classifier["weight_mean"], classifier["bias_mean"]
    # (f, num_classes), (num_classes, )
    weight_log_variance, bias_log_variance = (
        classifier["weight_log_variance"],
        classifier["bias_log_variance"],
    )
    # (b, num_classes,)
    logits_mean_test = tf.matmul(features_test, weight_mean) + bias_mean
    #                            (b, f) x (f, num_classes)     (num_classes,)
    # (b, num_classes,)
    logits_log_var_test = tf.math.log(
        tf.matmul(features_test ** 2, tf.exp(weight_log_variance))
        + tf.exp(bias_log_variance)
    )
    logits_std = tf.sqrt(tf.exp(logits_log_var_test))
    # (samples, b, num_classes,)
    logits_sample_test = tfp.distributions.Normal(
        loc=logits_mean_test, scale=logits_std
    ).sample(samples)
    # logits_sample_test = sample_normal(
    #     logits_mean_test, logits_log_var_test, samples
    # )
    # (samples, b, num_classes,)
    test_labels_tiled = tf.tile(tf.expand_dims(test_outputs, 0), [samples, 1, 1])
    # (samples, b)
    # task_log_py = multinoulli_log_density(
    #     inputs=test_labels_tiled, logits=logits_sample_test
    # )
    task_log_py = tfp.distributions.Multinomial(
        total_count=way, logits=logits_sample_test
    ).log_prob(tf.cast(test_labels_tiled, tf.float32))
    # (b, num_classes)
    averaged_predictions = tfp.math.reduce_logmeanexp(
        input_tensor=logits_sample_test, axis=0
    )
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
    task_score = tfp.math.reduce_logmeanexp(input_tensor=task_log_py, axis=0)
    # ()
    task_loss = -tf.reduce_mean(input_tensor=task_score, axis=0)

    return task_loss, task_accuracy


def main():
    args = parse_command_line()

    print("Options: %s\n" % args)

    # Load training and eval data
    data = get_data(args.dataset)

    model = Model(args.d_theta, True, args.dropout)

    # evaluation samples
    eval_samples_train = 15
    eval_samples_test = args.shot

    # testing parameters
    test_iterations = 600
    test_args_per_batch = 1  # always use a batch size of 1 for testing

    # average all values across batch

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    validation_batches = 20

    ckpt = tf.train.Checkpoint(step=tf.Variable(0), model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, args.checkpoint_dir, max_to_keep=3)
    # Main training loop

    def eval_task(inputs, training):
        return evaluate_task(
            model, inputs, samples=args.samples, way=args.way, training=training
        )

    @tf.function
    def eval_step(inputs, training=False):
        train_inputs, test_inputs, train_outputs, test_outputs = inputs
        batch_output = tf.map_fn(
            fn=functools.partial(eval_task, training=training),
            elems=(train_inputs, train_outputs, test_inputs, test_outputs),
            dtype=(tf.float32, tf.float32),
            parallel_iterations=args.tasks_per_batch,
        )
        batch_losses, batch_accuracies = batch_output
        iteration_loss = tf.reduce_mean(input_tensor=batch_losses)
        iteration_accuracy = tf.reduce_mean(input_tensor=batch_accuracies)
        return iteration_loss, iteration_accuracy

    @tf.function
    def train_step(inputs, training=True):
        train_inputs, test_inputs, train_outputs, test_outputs = inputs
        with tf.GradientTape() as tape:
            # tf mapping of batch to evaluation function
            batch_output = tf.map_fn(
                fn=functools.partial(eval_task, training=training),
                elems=(train_inputs, train_outputs, test_inputs, test_outputs),
                dtype=(tf.float32, tf.float32),
                parallel_iterations=args.tasks_per_batch,
            )
            batch_losses, batch_accuracies = batch_output
            iteration_loss = tf.reduce_mean(input_tensor=batch_losses)
            iteration_accuracy = tf.reduce_mean(input_tensor=batch_accuracies)
        grads = tape.gradient(iteration_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return iteration_loss, iteration_accuracy

    train_accuracy = tf.keras.metrics.Mean()
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    while int(ckpt.step.numpy()) < args.iterations:
        train_inputs, test_inputs, train_outputs, test_outputs = data.get_batch(
            "train", args.tasks_per_batch, args.shot, args.way, eval_samples_train
        )
        inputs = (train_inputs, test_inputs, train_outputs, test_outputs)
        iteration_loss, iteration_accuracy = train_step(inputs)
        train_accuracy.update_state(iteration_accuracy)

        # Validation
        if (int(ckpt.step) > 0) and (int(ckpt.step) % args.print_freq == 0):
            # compute accuracy on validation set
            validation_accuracy = tf.keras.metrics.Mean()
            validation_iteration = 0
            while validation_iteration < validation_batches:
                train_inputs, test_inputs, train_outputs, test_outputs = data.get_batch(
                    "validation",
                    args.tasks_per_batch,
                    args.shot,
                    args.way,
                    eval_samples_test,
                )
                inputs = (train_inputs, test_inputs, train_outputs, test_outputs)
                iteration_loss, iteration_accuracy = eval_step(inputs)
                validation_accuracy.update_state(iteration_accuracy)
                validation_iteration += 1
            print(
                "Iteration: {}, Loss: {:5.3f}, Train-Acc: {:5.3f}, Val-Acc: {:5.3f}".format(
                    int(ckpt.step),
                    iteration_loss,
                    train_accuracy.result().numpy(),
                    validation_accuracy.result().numpy(),
                )
            )
            train_accuracy.reset_states()

        ckpt.step.assign_add(1)
        if int(ckpt.step) % 10 == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

    def test_model():
        test_iteration = 0
        test_iteration_accuracy = []
        while test_iteration < test_iterations:
            train_inputs, test_inputs, train_outputs, test_outputs = data.get_batch(
                "test",
                test_args_per_batch,
                args.test_shot,
                args.test_way,
                eval_samples_test,
            )

            batch_output = tf.map_fn(
                fn=functools.partial(eval_task, training=False),
                elems=(train_inputs, train_outputs, test_inputs, test_outputs),
                dtype=[tf.float32, tf.float32],
                parallel_iterations=args.tasks_per_batch,
            )
            batch_losses, batch_accuracies = batch_output
            # iteration_loss = tf.reduce_mean(input_tensor=batch_losses)
            iter_acc = tf.reduce_mean(input_tensor=batch_accuracies)
            test_iteration_accuracy.append(iter_acc)
            test_iteration += 1
        test_accuracy = np.array(test_iteration_accuracy).mean() * 100.0
        confidence_interval_95 = (
            196.0 * np.array(test_iteration_accuracy).std()
        ) / np.sqrt(len(test_iteration_accuracy))
        print(
            "Held out accuracy: {0:5.3f} +/- {1:5.3f}".format(
                test_accuracy, confidence_interval_95
            )
        )
        print(
            "Train Shot: {0:d}, Train Way: {1:d}, Test Shot {2:d}, Test Way {3:d}".format(
                args.shot, args.way, args.test_shot, args.test_way
            )
        )

    # test the model on the final trained model
    # no need to load the model, it was just trained
    test_model()


if __name__ == "__main__":
    main()
