# What a fixed point would mean for Inkling

Weight sharing makes a fixed-point analysis possible. It does not show that the model reaches a fixed point.

## The state being repeated

Inkling has 24 stored layers. Layers 6–17 run three times.
At a pass boundary, it saves two six-layer output blocks.
The next pass restarts from the same prelude and reads messages from those saved blocks.
Each layer chooses separate messages for its attention query, key, value, and feed-forward input.
The [pinned implementation](https://github.com/huggingface/transformers/blob/6173a2f679ff4ed393afc499aefc569da7e68bc1/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py) defines this computation.

For a whole fixed prompt X, write the two saved blocks at all token positions as S_t.
With caching disabled, passes after initialization implement a shared map:

```math
S_{t+1}=F_X(S_t).
```

The first pass initializes this recurrence without a previous message bank.
The final six layers decode the third pass's blocks together with the retained prelude.
One token's bank alone is not the full state: attention also reads earlier token positions.
Cached generation additionally has a separate key/value cache for each layer visit.

An equilibrium is a state S* with F_X(S*)=S*.
The production model executes three passes rather than solving that equation.
Three observed passes cannot establish asymptotic convergence.

## Scale is a nearly invisible coordinate

The message reader first computes, separately for each saved block:

```math
n(b)=b/\sqrt{\operatorname{mean}(b^2)+\epsilon}.
```

For positive c and block variance much larger than epsilon, n(cb) is nearly n(b).
The exact discrepancy comes from epsilon and floating-point arithmetic.
This predicts that independently doubling either saved block barely changes the next pass's messages.
It is a forward-model prediction, not a lens assumption.

A meaningful convergence diagnostic should therefore inspect normalized block directions as well as raw distances.
It should also separate prompt-dependent content from any shared loop bookkeeping.
An activation trajectory can move far along coordinates that the next pass discards.

## Two different derivatives

Near an equilibrium, let A be the derivative of F with respect to S.
A one-time perturbation to the current state has a later effect proportional to A^k after k passes.
If the equilibrium is stable, this effect can decay.

By contrast, changing an input that drives every pass has equilibrium sensitivity:

```math
\frac{dS^*}{dX}=(I-A)^{-1}\frac{\partial F}{\partial X}.
```

These answer different questions. Applying the inverse to an instantaneous activation is not automatically a valid lens.
For Inkling's actual finite execution, use the product of derivatives along the actual remaining passes.
An infinite-horizon inverse needs additional stability and stationarity evidence.
See [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377) for implicit differentiation through a solved equilibrium.

## A concrete lens proposal

Start with the separately normalized bank pair at a loop boundary.
Fit a separate averaged Jacobian for each remaining loop horizon.
Keep source-token and target-token averaging explicit.
Compare this with maps that freeze routing coefficients during the backward pass.

For a message m and current input x, Inkling's gate is:

```math
u=W_x n(x)+W_m n(m),\qquad
y=(1-\tanh(Au))\odot x+\tanh(Bu)\odot m.
```

Holding x fixed, its ideal real-valued Jacobian is:

```math
\frac{\partial y}{\partial m}=\operatorname{diag}(g)+R,
\quad g=\tanh(Bu),\quad \operatorname{rank}(R)\leq64.
```

Thus rank-64 control does not imply rank-64 message transmission.
The diagonal path carries message content directly. The low-rank term changes the routing gains.
If all diagonal entries are nonzero, the total Jacobian has rank at least 2304-64=2240.
This is a local algebraic bound, not a measured semantic memory capacity.

Both proposed maps need held-out concept readouts and causal swaps.
Clearer words alone would not establish that either map explains the model's reasoning.
