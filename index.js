
let nn = new NeuralNetwork(3,2,2);

console.log('hidden_layer', nn.hidden_layer);
console.log('output_layer', nn.output_layer);

// let output = nn.feedforward([0.1, 0.5, 0.6]);
// console.log('output',output[0]);

let result = nn.train([0.1, 0.2, 0.3], [0.5, 1]);
// console.log('result', result);
