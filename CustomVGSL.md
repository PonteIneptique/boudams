# Custom Architecture Building String

*This is following the example of [VGSL specs](https://tesseract-ocr.github.io/tessdoc/tess4/VGSLSpecs.html).*

The new spec system is built around custom architecture strings.

Available modules:

- `C<x>,<d>[,<p>]` uses a convolutional layer where `x` is the n-gram window and `d` the output. `p` is an optional padding.
- `CS[s]<x>,<d>,<l>[,Do<r>]` uses a sequential convolutional layer where `x` is the "n-gram window", `d` the output, `l` the number. Can have an optional `Dropout` rate between each convolution. 
of layers. `[s]` (`CSs`) use a final addition of conved output + original input with a scale   
- `P[l]` adds a positional embeddings with an optional linear activation (eg. `Pl`).
- `L<h>,<l>` uses a Bi-LSTM layer where `h` is the hidden size and `l` the number of layers.
- `G<h>,<l>` uses a Bi-GRU layer where `h` is the hidden size and `l` the number of layers.
- `Do<r>` uses a Dropout layer with a rate of `r`
- `L<d>` uses a Linear layer of dimension `d`

The VGSL module must starts with an embedding size: `E<dim>`.

Example: `[E200 L120 L200 Cr3,10 D3]` will use a Convolutional Layer of (3 ngram for 10 of dim) and a relu activation
over which 30% of dropout is applied before classification

## Legacy architectures

- ConvPos `[E256 Pl Do.3 CS5,256,10,Do.25 L256]` 
- ConvNoPos `[E256 Do.3 CS5,256,10,Do.25 L256]` 
- Gru `[E256 Do.3 CSs5,256,10Do.25 L256]`
