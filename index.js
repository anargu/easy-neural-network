
let nn = new NeuralNetwork(2,2,1);

console.log('hidden_layer', nn.hidden_layer);
console.log('output_layer', nn.output_layer);

// let output = nn.feedforward([0.1, 0.5, 0.6]);
// console.log('output',output[0]);

// let result = nn.trainV2([0.1, 0.2, 0.3], [0.5, 1]);
// console.log('result', result);

let training_data = [
  {
    inputs: [0, 0],
    targets: [0]
  },
  {
    inputs: [1, 0],
    targets: [1]
  },
  {
    inputs: [0, 1],
    targets: [1]
  },
  {
    inputs: [1, 1],
    targets: [0]
  }
]

// nn.trainV2([1,1],[0]);
// nn.trainV2([0,0],[0]);
// nn.trainV2([0,1],[1]);
// nn.trainV2([1,0],[1]);

for(let i = 0; i < 10000; i++) {
  // let data = random(training_data);
  // nn.trainV2(data.inputs, data.targets);
  for (data of training_data) {
    nn.trainV2(data.inputs, data.targets);
  }
}

let result = nn.feedforward([0,0]);
let result1 = nn.feedforward([1,0]);
let result2 = nn.feedforward([0,1]);
let result3 = nn.feedforward([1,1]);
console.log('result [0,0]',result[0]._data[0][0]);
console.log('result1 [1,0]',result1[0]._data[0][0]);
console.log('result2 [0,1]',result2[0]._data[0][0]);
console.log('result3 [1,1]',result3[0]._data[0][0]);
