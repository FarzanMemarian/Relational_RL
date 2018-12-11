# relational-RL

## Case 1
- the most basic case
- uses keras on top of tensorflow
- objects are removed after they are reached
- all fully connected layers, no convolution
- no relational-module used
- HRL is implemented based on Kulhani's paper

## Case 2
- the first version on Pytorch
- no relational module is used, but the input is passed as a tensor to the controller and meta controller rather than a vector
- input to the meta-controller is just the current matrix representing the gridworld where -1 is added to the current location that the agent is occupying. 
- input to the controller is a little triciker, it's the same as the input to the meta-controller but we need to pass in the goal as well. There are multiple ways to do this, one is to add another channel of the same dimension as the gridworld matrix where each entry is the current goal. 

## Case 3
- relational module is being implemented
