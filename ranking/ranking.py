import tensorflow as tf
import tensorflow_ranking as tfr
import numpy as np
from tensorflow.contrib.learn import ModeKeys

from tensorflow.python.ops import parsing_ops

from absl import flags

flags.DEFINE_string("train_path", None, "Input file path used for training.")
flags.DEFINE_string("vali_path", None, "Input file path used for validation.")
flags.DEFINE_string("test_path", None, "Input file path used for testing.")
flags.DEFINE_string("output_dir", None, "Output directory for models.")
flags.DEFINE_string("embedding_path", None, "Path to npy embedding matrix")

flags.DEFINE_integer("train_batch_size", 64, "The batch size for training.")
flags.DEFINE_integer("num_train_steps", 100000, "Number of steps for training.")

flags.DEFINE_float("learning_rate", 0.01, "Learning rate for optimizer.")
flags.DEFINE_float("dropout_rate", 0.5, "The dropout rate before output layer.")
flags.DEFINE_list("hidden_layer_dims", ["512", "256", "128"],
                  "Sizes for hidden layers.")

flags.DEFINE_integer("list_size", 100, "List size used for training.")
flags.DEFINE_integer("group_size", 1, "Group size used in score function.")

flags.DEFINE_string("loss", "pairwise_logistic_loss",
                    "The RankingLossKey for loss function.")

FLAGS = flags.FLAGS
tf.enable_eager_execution()


class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialize data iterator after session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_fn = None

    def after_create_session(self, session, coord):
        """Initialize the iterator after the session has been created."""
        del coord
        self.iterator_initializer_fn(session)


class HeadWithScaffold(tfr.head._RankingHead):
    def __init__(self, *args, **kwargs):
        self.scaffold = kwargs.pop('scaffolding', '')
        super(HeadWithScaffold, self).__init__(*args, **kwargs)

    def create_estimator_spec(self,
                              features,
                              mode,
                              logits,
                              labels=None,
                              regularization_losses=None):
        estimator = super(HeadWithScaffold, self).create_estimator_spec(
            features, mode, logits, labels, regularization_losses
        )
        if not self.scaffold or mode != ModeKeys.TRAIN:
            return estimator

        return estimator._replace(scaffold=self.scaffold)


def make_score_fn():
    """Returns a groupwise score fn to build `EstimatorSpec`."""

    def _score_fn(context_features, group_features, mode, params, config):
        """Defines the network to score a group of documents."""

        emb_placeholder = tf.placeholder(dtype=tf.float32,
                                         shape=params['embedding_shape'],
                                         name='embedding_placeholder')
        emb = tf.get_variable('embedding',
                              dtype=tf.float32,
                              initializer=emb_placeholder,
                              trainable=False)

        with tf.name_scope("input_layer"):
            query = context_features['query']
            query_layer = tf.nn.embedding_lookup(emb, query)
            query_layer = tf.reduce_max(query_layer, 1)

            candidates = tf.squeeze(group_features['candidates'], -1)
            candidates_layer = tf.nn.embedding_lookup(emb, candidates)
            candidates_layer = tf.reshape(candidates_layer, (
                candidates.shape[0], -1
            ))

            input_layer = tf.concat(
                [query_layer, candidates_layer], 1
            )
            # tf.summary.scalar("input_sparsity", tf.nn.zero_fraction(input_layer))
            # tf.summary.scalar("input_max", tf.reduce_max(input_layer))
            # tf.summary.scalar("input_min", tf.reduce_min(input_layer))

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        cur_layer = tf.layers.batch_normalization(input_layer, training=is_training)
        for i, layer_width in enumerate(int(d) for d in FLAGS.hidden_layer_dims):
            cur_layer = tf.layers.dense(cur_layer, units=layer_width)
            cur_layer = tf.layers.batch_normalization(cur_layer, training=is_training)
            cur_layer = tf.nn.relu(cur_layer)
            tf.summary.scalar("fully_connected_{}_sparsity".format(i),
                              tf.nn.zero_fraction(cur_layer))
        cur_layer = tf.layers.dropout(
            cur_layer, rate=FLAGS.dropout_rate, training=is_training)
        logits = tf.layers.dense(cur_layer, units=FLAGS.group_size)
        return logits

    return _score_fn


def get_eval_metric_fns():
    """Returns a dict from name to metric functions."""
    metric_fns = {}
    # metric_fns.update({
    #     "metric/%s" % name: tfr.metrics.make_ranking_metric_fn(name) for name in [
    #         tfr.metrics.RankingMetricKey.ARP,
    #         tfr.metrics.RankingMetricKey.ORDERED_PAIR_ACCURACY,
    #     ]
    # })
    metric_fns.update({
        "metric/nDCG@%d" % topn: tfr.metrics.make_ranking_metric_fn(
            tfr.metrics.RankingMetricKey.NDCG, topn=topn)
        for topn in [5, 10, 20, 50]
    })
    metric_fns.update({
        "metric/mrr@%d" % topn: tfr.metrics.make_ranking_metric_fn(
            tfr.metrics.RankingMetricKey.MRR, topn=topn)
        for topn in [5, 10, 20, 50]
    })
    return metric_fns


def get_train_input(tf_records, batch_size, list_size):
    iterator_initializer_hook = IteratorInitializerHook()

    def _train_input_fn():
        ds = tfr.data.read_batched_sequence_example_dataset(
            tf_records,
            batch_size,
            list_size,
            context_feature_spec={
                "query": parsing_ops.FixedLenFeature([5], tf.int64)
            },
            example_feature_spec={
                "candidates": parsing_ops.FixedLenFeature([1], tf.int64,
                                                          default_value=tf.constant([-1], tf.int64)),
                "relevance": parsing_ops.FixedLenFeature([1], tf.int64,
                                                         default_value=tf.constant([0], tf.int64))
            },
            reader_args=['GZIP']
        )

        ds = ds.map(lambda f: (f, tf.cast(tf.squeeze(f.pop('relevance'), -1), tf.float32)))
        iterator = ds.make_initializable_iterator()
        iterator_initializer_hook.iterator_initializer_fn = \
            lambda sess: sess.run(iterator.initializer)
        return iterator.get_next()

    return _train_input_fn, iterator_initializer_hook


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    # embedding = np.zeros((5500000, 16))
    # embedding = np.vstack((embedding, np.zeros((1, embedding.shape[1]))))

    embedding = np.load(FLAGS.embedding_path)

    def _train_op_fn(loss):
        """Defines train op used in ranking head."""
        return tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=FLAGS.learning_rate,
            optimizer="Adagrad")

    ranking_head = HeadWithScaffold(
        loss_fn=tfr.losses.make_loss_fn(FLAGS.loss),
        eval_metric_fns=get_eval_metric_fns(),
        optimizer=None,
        train_op_fn=_train_op_fn,
        scaffolding=tf.train.Scaffold(
            init_feed_dict={
                'groupwise_dnn_v2/group_score/embedding_placeholder:0': embedding
            }
        ),
        name=None)

    serving_input = tfr.data.build_sequence_example_serving_input_receiver_fn(
        100,
        context_feature_spec={
            "query": parsing_ops.FixedLenFeature([5], tf.int64)
        },
        example_feature_spec={
            "candidates": parsing_ops.FixedLenFeature([1], tf.int64,
                                                      default_value=np.array([-1], dtype=np.int64)),
        }
    )

    estimator = tf.estimator.Estimator(
        model_fn=tfr.model.make_groupwise_ranking_fn(
            group_score_fn=make_score_fn(),
            group_size=FLAGS.group_size,
            transform_fn=lambda f: ({'query': f.pop('query')}, f),
            ranking_head=ranking_head),
        config=tf.estimator.RunConfig(
            FLAGS.output_dir, save_checkpoints_steps=1000),
        params=dict(
            embedding_shape=embedding.shape
        )
    )

    train_input_fn, train_hook = get_train_input(FLAGS.train_path,
                                                 FLAGS.train_batch_size,
                                                 FLAGS.list_size)
    vali_input_fn, vali_hook = get_train_input(FLAGS.vali_path,
                                               FLAGS.train_batch_size,
                                               FLAGS.list_size)
    # test_input_fn, test_hook = get_eval_input()

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        hooks=[
            train_hook,
        ],
        max_steps=FLAGS.num_train_steps)

    def _best_model(best_eval_result, current_eval_result):
        return current_eval_result['metric/nDCG@50'] > best_eval_result['metric/nDCG@50']

    vali_spec = tf.estimator.EvalSpec(
        input_fn=vali_input_fn,
        hooks=[vali_hook],
        steps=100,
        exporters=[
            tf.estimator.BestExporter(
                serving_input_receiver_fn=serving_input,
                compare_fn=_best_model
            )
        ],
        start_delay_secs=0,
        throttle_secs=30)

    # Train and validate
    tf.estimator.train_and_evaluate(estimator, train_spec, vali_spec)

    # Evaluate on the test data.
    # estimator.evaluate(input_fn=test_input_fn, hooks=[test_hook])


if __name__ == '__main__':
    flags.mark_flag_as_required("train_path")
    flags.mark_flag_as_required("vali_path")
    # flags.mark_flag_as_required("test_path")
    flags.mark_flag_as_required("output_dir")

    tf.app.run()
