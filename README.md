## GraphSage: Representation Learning on Large Graphs

#### Authors: [William L. Hamilton](http://stanford.edu/~wleif) (wleif@stanford.edu), [Rex Ying](http://joy-of-thinking.weebly.com/) (rexying@stanford.edu)
#### [Project Website](http://snap.stanford.edu/graphsage/)

**Attention! This is modified version of original codebase provided by authors of the paper.**

I introduced next improvements:
* networkx is replaced with graph-tool* (allowed fast graph loading from binary file; faster dataset building for unsupervised training)
* train/test split is now happening on the fly for both supervised/unsupervised setups (no need to label nodes in advance)
* adjacency matrix is now being cached (speed-up setup before training)
* fixed issue with features matrix > 2Gb
* models are now stored as TF-saved_model (in the end of training for unsupervised, best model on f1-micro for supervised)
* inference is moved out
* two fully-connected layers were added after convolutions according to Pinterest paper (
[Graph Convolutional Neural Networks for Web-Scale Recommender Systems](https://arxiv.org/abs/1806.01973) by Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai, William L. Hamilton, Jure Leskovec
)
Kudos to authors of original code!

*graph-tool installation:
```
https://git.skewed.de/count0/graph-tool/wikis/installation-instructions
```

