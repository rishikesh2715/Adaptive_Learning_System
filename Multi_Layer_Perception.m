FEEDFORWARD NEURAL NETWORK
To create a feedforward backpropagation network we can use NEWFF 

   Syntax

   net = newff(PR,[S1 S2...SNl],{TF1 TF2...TFNl},BTF,BLF,PF) 
   Description

       NEWFF(PR,[S1 S2...SNl],{TF1 TF2...TFNl},BTF,BLF,PF) takes,
         PR  - Rx2 matrix of min and max values for R input elements.
         Si  - Size of ith layer, for Nl layers.
         TFi - Transfer function of ith layer, default = 'tansig'.
         BTF - Backprop network training function, default = 'trainlm'.
         BLF - Backprop weight/bias learning function, default = 'learngdm'.
         PF  - Performance function, default = 'mse'.
       and returns an N layer feed-forward backprop network.

Consider this set of data:
p=[-1 -1 2 2;0 5  0 5]
t =[-1 -1 1 1]
where p is input vector and t is target.

Suppose we want to create feed forward neural net with one hidden layer,  3 nodes in hidden layer, with tangent sigmoid as transfer function in hidden layer and linear function for output layer, and with gradient descent with momentum backpropagation training function, just simply use the following commands:

» net=newff([-1 2;0 5],[3 1],{'tansig' 'purelin'},’traingdm’);

Note that the first input [-1 2;0 5] is the minimum  and maximum values of vector p. We might use minmax(p) , especially for large data set, then the command becomes:

»net=newff(minmax(p),[3 1],{'tansig' 'purelin'},’traingdm’);