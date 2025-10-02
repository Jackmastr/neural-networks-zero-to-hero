My implementation work for Andrej Karpathy's Neural Networks course: https://www.youtube.com/watch?v=VMj-3S1tku0&amp;list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ

## 1 Micrograd
### First Watch (2-3x speed)
- Micrograd is all that's needed for NN - rest is optimization
- He creates a "Value" object - but why not just use the numbers
- OK bc there is a graph structure with children, etc.
- each node has the gradient wrt its parents?
- by gradient we mean the definition of the derivative: f(x) + f(x+h) / h
- Actually maybe by gradient we ACTUALLY mean that we don't have to nudge it by h.... if the individual neuron is only doing something simple that we already know analytically what the derivative is! Looks like HERE at least its just arithmetic or even the activations are simple enough
- chain rule
- does the node only save its derivative wrt to the output actually?
- topological sort puts them all in order from left to right, why do you not start out with this or why need to order at all?
- keep in mind if one node is used in multiple expressions when calculating the derivative
- The NN part seemed particularly confusing to me so I'd like to slow that part down I think and pay more attention and check my understanding more
- Loss lets you take problem with all your "final outputs" and turns it all into 1 number you can do gradient descent to make better
- Common mistake is to not "zero grad" --- we have "+=" everywhere when calculating the gradient so when we recalculate it after taking a step the previous gradients are contributing when we don't want them too 
### Main Watch
- want derivatives wrt weights and not data because the data is fixed and the weights are what we have control over in our NN
- grad is derivative of OUTPUT wrt that value
- thinking about the backpropagation ordering... we always want to work backward. We want to update all the nodes once and only once + always update a node AFTER the later gradients are updated already
- The above criteria can be achieved via topological sort --> produces a DAG, all the edges only go 1 way.
- ~~The edges here actually go OPPOSITE how I would think. This is because we are thinking of the right order for backpropagation not the forward propagation. We start at the terminal node. Recursively call backwards and only add the current node to the list after all of the previous ones have been added~~
- actually it goes in the right order we just reverse it.
- First implementation assumes each variable is used only once. BOTH in individual expressions (like a+a) or across multiple expressions (like e=a+b, d=a x b, f = e+d). Need to sum contributions.
- rmul used if mul fails
- As long as you can do forward and backward pass (read: you can compute the local derivative), each component can be as complicated or atomic as you want
- [a].item() returns a in pytorch 
- zip(a, b) returns iterator over  ((a1, b1) (a2, b2), ...) pairing up elements of a and b
- When doing the backward need to zero the gradient of all the nodes except the final one which has a grad of 1
- when there a huge number of examples you might take a random subset, a BATCH, and do the forward pass --> backward pass --> update based on the gradients and step size loop on that batch
- In pytorch you can define your own building block for a neuron function (like add, multiply, tanh, power, legendre polynomial) all you need is the function (forward pass) and its derivative (backward pass)!
### Exercises
- used gdown to download the collab (thanks chatgpt)
- Tried to implement division without defining the power first and I didn't think it through. For a/b deriv wrt a is 1/b but deriv wrt b is NOT a ... it's -a/b^2 !
## 2 Makemore
### Main Watch
- used wget to get the text file
- each name has a lot of info. Eg. start with I, after I there is S, after ISABELLA it's likely to end, etc.
- A bigram only looks at pairs of letters, here for the model it's what letter or (end of word) follows the current letter
- lets count how often each pair arrives (adding special chars for beginning and end)
- store counts for each pair as an NxN tensor of dtype int32 not float
- .item() will pop out the individual element instead of a tensor of 1 element
- GENERATOR makes sampling the probability distribution DETERMINISTIC
- MULTINOMIAL gets samples given a probability distribution
- efficiency idea --> upfront calculate the normalized probabilities for each row (i.e. each row is a first letter, get odds for second given first)
- dim=0 is rows, but so torch.sum(dim=0) travels along the rows and sums the columns? still a bit confused
- OK I think I see. summing over dim 0 does "sum the rows" in that you add each row together ``` [[1,2,3],[4,5,6]] --> [5,7,9]
- https://docs.pytorch.org/docs/stable/notes/broadcasting.html broadcasting still kinda mixes me up --> start at the ending dimension, it should be equal or 1 or DNE
- Broadcasting adds copies where the dimensions are just 1 until there is enough to match the other one. Once it's all done then it does the operation elementwise on the expanded array(s)
- Wow! He says to "really respect" and "really understand" broadcasting. To even look up some tutorials haha.
- Oh wow, especially with the square tensor watch out. Without keepdim=True it's not (27x27) divide out (27x1 --> expanded to 27x27 by copying the row) it's (27x27) divide out (27), it thinks the columns align cause it starts from the last dim, then it expands out to fill the rows which is wrong. IT IS EASY TO MIS-BROADCAST!
- In-place is much faster: P /= P.sum() is better than P = P / P.sum()
- prob is [0,1], logprob is [-inf,0]
- print(f'{x=})
- "Smooth" model by adding 1 (or small number) to every count so there is at least a nonzero probability of everything, you avoid infinities in -log-likelihood
- torch.tensor vs torch.Tensor LOL the actual difference is tensor infers type, Tensor defaults to float32!
- the legendary **ONE-HOT ENCODING**! It's how you can encode an integer as input to a NN! 0=[1 0 0], 1=[0, 1, 0], 2=[0, 0, 1], etc!
- torch.nn.functional.one_hot lol
- huh one_hot preserves the dtype so you need to cast as float
- torch.randn draws from Normal Distro
- LOGITS are sort of like log(counts) or log(odds) or similar...
- W.grad = None is more efficient than W.grad = 0 for doing the zero_grad
- remember to add requires_grad=True when creating W, by default it doesn't as a time-saving thing it assumes it's a leaf node
- You can incentivize the weights to be more equal if you want the equivalent of SMOOTHING from earlier
- incentivize in the LOSS FUNCTION --> REGULARIZATION!

## 3 MLP
- In general you use words/tokens instead of characters alone. And you put those words in a vector space where nearby words mean similar things so you can make connections/substitutions
- THE SPACE IS MUCH SMALLER THAN THE NUMBER OF WORDS of course!
- one_hot give a long not a float
- one_hot @ C takes the input word and plucks out the row corresponding to the vector of that word (here 27 chars, for each char there is a 2d vector associated with it), since one hot is just like delta_ij
- you can just index you don't need one_hot
- C\[\[5,6,7\]\] or x=tensor(1,2,3,2,2,3) and do C\[x\]
- `print(C[X][13,2])=print(C[1]) # why are they equal? Well X[13,2] = 1 and C[X]_ij=C[X[i,j]] so C[X]_13,2 = C[X[13,2]] = C[1] ok that's a little trippy`
- Woah views seem powerful! They seem to reshape the tensor but don't change underlying attributes so it's very powerful
- In memory all the tensors are just 1d lists under the hood
- how pytorch works under the hood https://blog.ezyang.com/2019/05/pytorch-internals/
- a.view(-1, 6) will infer the right number for -1 automagically
- CONCATENATION IS INEFFICIENT
- cross_entropy does the softmax then -loglikelihood calculation all at once
- cross entropy is also much more efficient with both forward and backward passes where things are optimized to do it all together in 1 step. (fused kernel and analytic derivative can be simple anyways)
- can be more numerically well-behaved if only 1 step, underflow/overflow/precision loss over steps that involve exp etc. MAINLY OVERFLOW at least in the example given
- LOGITS ARE INSENSITIVE TO A GLOBAL SHIFT, so because exp^big is bad, under the hood pytorch subtracts off by the max of the tensor then does the softmax
- You batch because doing forward and backward on the whole data can be really slow
- Technically then the gradient is just an approximation but it is good enough for government work most of the time and it's so much faster it's OK to have to take more steps in response
- How to pick a good learning rate? Train while trying new learning rate every step and then plot the likelihood and look for a minimum
- LEARNING RATE DECAY --> towards the end make the learning rate smaller to see if it can improve just a bit more
- 80/10/10 train/dev/test set. dev for hyperparameters
- If loss is about the same on train and dev that means you are UNDERFITTING, you can squeeze more juice by increasing the size of the model until you overfit then you've gone too far
- Hyperparameter tuning ideas: 
	- Number of neurons in hidden layer
	- Dimensionality of the embedding
	- Number of characters that are part of the context
	- Number of steps
	- Learning rate
	- Learning rate over time (learning rate decay)
	- Batch size (more for how long to change the model)
	- Regularization?
### A Neural Probabilistic Language Model (Bengio et al. 2003) Notes
- Necessary to move beyond n-grams as you go to longer contexts
- If words are represented as vectors then it can, in novel situations, take advantage of what it has learned about related words
#### 1 Introduction
- 10 words in vocab of 10K = 10^50 permutations
- For continuous functions you take advantage of ~local smoothness, for discrete there is something similar though
- Think conditional prob of next word given all before in context
- 2 issues w ngrams --> what to do if not seen in training data? and obviously more context would be really useful
- Summary of approach in 3 bullets:
	- 1. associate each word with a vector
	- 2. joint prob function of sequences of words in terms of these vectors
	- 3. learn simultaneously the vectors and prob function params
- ![[Pasted image 20250810213839.png]]
#### 2 A Neural Model
- **perplexity:** = exp(avg(-LL))
- 
