const LR = 0.1;

class NeuralNetwork {
  constructor(i_nodes, h_nodes, o_nodes) {
    this.input_nodes = i_nodes + 1;
    this.hidden_nodes = h_nodes + 1;
    this.output_nodes = o_nodes;


    this.hidden_layer = this.createWMatrix(this.input_nodes, this.hidden_nodes);
    this.output_layer = this.createWMatrix(this.hidden_nodes, this.output_nodes);
  }

  createWMatrix(rows, cols) {
    let mattr = [];
    for (let i = 0; i < rows; i++) {
      mattr[i] = [];
      for (let j = 0; j < cols; j++) {
        mattr[i][j] = math.random(-1, 1);
      }
    }
    mattr = math.matrix(mattr);
    return math.transpose(mattr);
  }

  // HxI - Ix1 => Hx1
  // OxH - Hx1 => Ox1
  // inputs => [..input..]
  feedforward(inputs) {
    inputs = inputs.concat([1]);
    let I = math.matrix([inputs]);
    I = math.transpose(I);
    const H = this.hidden_layer;

    const H1 = NeuralNetwork.activation(math.multiply(H, I));

    const O = this.output_layer;
    const O1 = NeuralNetwork.activation(math.multiply(O, H1));
    return [O1, H1];
  }

  train(inputs, targets) {
    let _inputs = inputs.concat([1]);
    targets = math.transpose(math.matrix([targets]));
    let I = _inputs;
    let [O, H] = this.feedforward(inputs);
    // let H = outputs[1];
    // let O = outputs[0];
    console.log('outputs', O);
    // compute the error
    let output_errors = math.subtract(targets, O);

    const HOT = math.transpose(this.output_layer);
    const hidden_errors = math.multiply(HOT, output_errors);

    // gradient
    let gradient = O.map(NeuralNetwork.dsigmoid);
    gradient = math.multiply(gradient, output_errors);
    gradient = math.multiply(gradient, LR);
    // deltas
    let ho_deltas = math.multiply(gradient, math.transpose(H));
    this.output_layer = math.add(this.output_layer, ho_deltas);

    // hidden gradient
    let hidden_gradient = H.map(NeuralNetwork.dsigmoid);
    hidden_gradient = math.multiply(hidden_gradient, hidden_errors);
    hidden_gradient = math.multiply(hidden_gradient, LR);
    // hidden deltas
    let ih_deltas = math.multiply(hidden_gradient, I);
    math.add(this.hidden_layer, ih_deltas);

    // console.log('output_errors', output_errors);
    // console.log('hidden_errors', hidden_errors);
  }

  static activation(mattr) {
    return mattr.map(function (value, index, matrix) {
      return NeuralNetwork.sigmoid(value);
    });
  }

  static sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  static dsigmoid(y) {
    // return 1 / (1 + Math.exp(-x));
    return y * (1 - y);
  }

}
