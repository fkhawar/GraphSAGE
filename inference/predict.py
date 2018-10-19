import sys
import numpy as np

from pyspark.sql import SparkSession
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.saved_model import loader


class Predict:
    session = None

    def __init__(self, model_path,
                 id_key,
                 meta_tag='serve',
                 meta_signature='predict',
                 meta_predictions='predictions'
                 ):
        self.model_path = model_path
        self.id_key = id_key
        self.meta_tag = meta_tag
        self.meta_signature = meta_signature
        self.meta_predictions = meta_predictions

    def __call__(self, inputs):
        if not self.session:
            self.graph = ops.Graph()
            with self.graph.as_default():
                self.session = tf.Session()
                metagraph_def = loader.load(
                    self.session, {self.meta_tag}, self.model_path)
            signature_def = metagraph_def.signature_def[self.meta_signature]

            # inputs
            self.feed_tensors = {
                k: self.graph.get_tensor_by_name(v.name)
                for k, v in signature_def.inputs.items()
            }

            # outputs/predictions
            self.fetch_tensors = {
                k: self.graph.get_tensor_by_name(v.name)
                for k, v in signature_def.outputs.items()
            }

        # Create a feed_dict for a single element.
        feed_dict = {
            tensor: [i[key] for i in inputs]
            for key, tensor in self.feed_tensors.items()
        }

        results = self.session.run(self.fetch_tensors, feed_dict)[self.meta_predictions]
        return [(i[self.id_key], res.tolist()) for i, res in zip(inputs, results)]


def main(model_path, src, dst, features_npy, batch_size=512):
    spark = SparkSession.builder.appName('SparkTensorflow').getOrCreate()

    rdd = spark._sc.textFile(src)
    count = rdd.count()

    features = spark._sc.broadcast(np.load(features_npy))

    predict = Predict(model_path=model_path, id_key='batch1')

    def prepare_input(node):
        return dict(
            batch_1=node,
            batch_2=node,
            features=features,
            neg_samples=0,
            batch_size=batch_size,
            dropout=0
        )

    rdd.map(prepare_input).repartition(count / batch_size + 1)\
        .mapPartitions(predict).saveAsTextFile(dst)


if __name__ == '__main__':
    main(*sys.argv[1:])
