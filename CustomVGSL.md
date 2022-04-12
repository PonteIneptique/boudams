# Custom Architecture Building String

*This is following the example of [VGSL specs](https://tesseract-ocr.github.io/tessdoc/tess4/VGSLSpecs.html).*

The new spec system is built around custom architecture strings.

Available modules:

- `C[A]<x>,<d>` uses a convolutional layer where `x` is the n-gram window and `d` the output.
- `CP[A]<x>,<d>` uses a convolutional layer with positional embeddings where `x` is the n-gram window and `d` the output.
- `L[A]<h>,<l>` uses a Bi-LSTM layer where `h` is the hidden size and `l` the number of layers.
- `G[A]<h>,<l>` uses a Bi-GRU layer where `h` is the hidden size and `l` the number of layers.
- `D<r>` uses a Dropout layer with a rate of `r`
- `L<d>` uses a Linear layer of dimension `d`

`[A]` can be replaced with an activation layer, such as:

- `s` = sigmoid
- `t` = tanh
- `r` = relu
- `l` = linear (i.e., No non-linearity)
- `m` = softmax
- `n` = n/a

The VGSL module must starts with an embedding size: `E<dim>`.

Example: `[E200 L120 L200 Cr3,10 D3]` will use a Convolutional Layer of (3 ngram for 10 of dim) and a relu activation
over which 30% of dropout is applied before classification
