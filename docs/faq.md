# FAQs

**Q:** If the weight of a conv layer is zero, the gradient will also be zero, and the network will not learn anything. Why "zero convolution" works?

**A:** This is wrong. Let us consider a very simple 

$$y=wx+b$$

and we have 

$$\partial y/\partial w=x, \partial y/\partial x=w, \partial y/\partial b=1$$

and if $w=0$ and $x \neq 0$, then 

$$\partial y/\partial w \neq 0, \partial y/\partial x=0, \partial y/\partial b\neq 0$$

which means as long as $x \neq 0$, one gradient descent iteration will make $w$ non-zero. Then 

$$\partial y/\partial x\neq 0$$

so that the zero convolutions will progressively become a common conv layer with non-zero weights.
