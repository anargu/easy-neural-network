const LR = 0.1;

function checkNaN(mattr) {
    // math.isNaN(this.hidden_layer)
    mattr.forEach((item) => {
      if(item === true){
        return true
      }
    });
    return false;
}

class NeuralNetwork {
  constructor(i_nodes, h_nodes, o_nodes) {
    this.input_nodes = i_nodes;
    this.hidden_nodes = h_nodes;
    this.output_nodes = o_nodes;


    this.ih_bias = this.createWMatrix(1, this.hidden_nodes);
    this.ho_bias = this.createWMatrix(1, this.output_nodes);
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
  feedforward(inputs, extended=false) {
    let I = math.matrix([inputs]);
    I = math.transpose(I);
    const H = this.hidden_layer;
    const BI = this.ih_bias;

    let hib = math.multiply(H, I);

    hib = math.add(hib, BI);

    const H1 = NeuralNetwork.activation(hib);

    const O = this.output_layer;
    let Z = math.multiply(O, H1);
    const BH = this.ho_bias;
    Z = math.add(Z, BH);
    const O1 = NeuralNetwork.activation(Z);
    if(extended === true) {
      return [O1, Z, O, H1, hib, H];
    } else {
      return [O1, H1];
    }
  }

  trainV2(inputs, targets) {
    let [O1, Z, O, H1, Zi, H] = this.feedforward(inputs, true);
    const I = math.matrix([inputs]);
    const target = math.transpose(math.matrix([targets]));
    let o_errors = math.subtract(target, O1);
    let dz = Z.map(NeuralNetwork.dsigmoid);

    const delta = math.dotMultiply(dz, o_errors);
    // delta 2x1
    // H1 2x1
    let grad_desc = math.multiply(delta, math.transpose(H1))
    let grad_desb_b = delta;

    grad_desc = math.multiply(grad_desc, LR);
    grad_desb_b = math.multiply(grad_desb_b, LR);

    // console.log('grad_desc', grad_desc);

    this.output_layer = math.add(this.output_layer, grad_desc);
    this.ho_bias = math.add(this.ho_bias, grad_desb_b);

    let deltaW = math.multiply(delta, this.output_layer);
    let dzi = Zi.map(NeuralNetwork.dsigmoid);
    const deltaI = math.dotMultiply(math.transpose(dzi), deltaW);

    let grad_desc_h = math.multiply(math.transpose(deltaI), I);
    let grad_desb_hb = math.transpose(deltaI);
    grad_desc_h = math.multiply(grad_desc_h, LR);
    grad_desb_hb = math.multiply(grad_desb_hb, LR);

    this.hidden_layer = math.add(this.hidden_layer, grad_desc_h);
    if(checkNaN(math.isNaN(this.hidden_layer))) {
      throw new Error(`item is NaN ${item}`);
    }

    this.ih_bias = math.add(this.ih_bias, grad_desb_hb);
    if(checkNaN(math.isNaN(this.hidden_layer))) {
      throw new Error(`item is NaN ${item}`);
    }

    // console.log('output_layer', this.output_layer);
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
