import argparse
import os
import numpy as np
from tensorflow.contrib.predictor import from_saved_model


def main(model_path, data_dir, output_name, chunk_size=512):
    predictor = from_saved_model(model_path)
    path = data_dir
    id_map = [int(v.strip()) for v in open(path + "id_map")]

    if os.path.isfile(path + "test-adj.npy"):
        sess = predictor.session
        op = sess.graph.get_operation_by_name('up_adj')
        adj_placeholder = sess.graph.get_tensor_by_name('Placeholder:0')
        sess.run(op, {
            adj_placeholder: np.load(path + "test-adj.npy")
        })

    emb = np.zeros((len(id_map), 256))
    for i in range(0, emb.shape[0], chunk_size):
        idx = list(range(i, min(i + chunk_size, len(id_map) - 1)))
        emb[i:i + len(idx)] = predictor({"batch_size": len(idx), "nodes": idx})['embeddings']

    np.save(path + output_name, emb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--chunk', type=int, default=512)

    args = parser.parse_args()

    main(args.model, args.data.rstrip('/') + '/', args.output, chunk_size=args.chunk)
