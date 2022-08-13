# Grokking JAX: How to get started with Deep Learning in Jax

If you have been developing your own deep learning models, you are probably familiar with the most popular deep learning frameworks out there: PyTorch and Tensorflow. However there is a new kid on the block and that is JAX. If you are still deciding whether JAX is right for you, check out my other blogpost _[5 Reasons why you should choose JAX for your deep learning projects instead of PyTorch or Tensorflow]()_.

## JAX a.k.a. numpy on steriods

So you have decided to explore JAX to see if it is something for you. Great! You open your browser, go to the official JAX documentation to better understand what JAX is. You are probably expecting something in the lines of:

_JAX: An open source machine learning framework that accelerates the path from research prototyping to production deployment._

But when you land on the homepage of JAX, well read for yourself...

_JAX is autograd and XLA brought together for high-performance numerical computing and machine learning research. It provides composable transformations of Python+NumPy programs: differentiate, vectorize, parallelize, Just-In-Time compile to GPU/TPU._

That's quite a lot to take in all in once. So lets dissect it. Essentially it is saying, JAX take care of computating the gradients for me so that I can train my machine learning Model. Ooh, and while we are at it, my models are quite big so make sure that I can ran them on any compute I want, whether that's CPUs, GPUs or even TPUs in a highly efficient way. That sounds better! So now what? Teach me the basics. Or better yet, show me some code!

## JAX: not your typical way to program

Before we really get started, we must understand one of the most important attributes of JAX, and that is: JAX only works with pure functions.

So what are pure functions? quite simple. It are functions with no side effects. Basically, these functions are idompotent, meaning they always have to same effect under the same conditions. So lets see an example of what would work and what would not work.

```
from jax import jit

n_pies = 0
def add_pies(pies_to_add: int):
    return n_pies + pies_to_add
print(jit(add_pies)(10))
>>> 10
```

Awesome 0 + 10 = 10, that's what we expected!
But now lets see what happens if we run the function again but change the global variable n_pies:

```
n_pies = 20
print(jit(add_pies)(10))

>>> 10
```

Darn... That doesn't seem right. The last time I checked 10 + 20 should have been 30, but we are still getting our old result back?! What's going on?

What is happening is that JAX first compiles your code and then runs it. Once it is run the second time it uses a cached compilation of your function which causes your results to be off... So what would a pure function look like? Like this:

```
from jax import jit

def pure_add_pies(n_pies: int, pies_to_add: int):
    return n_pies + pies_to_add

print(jit(pure_add_pies)(0, 10))
>> 10
print(jit(pure_add_pies)(20, 10))
>> 30
```

That's more like it! Let's keep that in mind, JAX compiles your functions and uses the cached version in subsequent calls to speed up computations. As a result, we need to make sure our punctions are pure :).

# Immutability

JAX has the attribute of being immutable and works with pure functions (more on that later). The reason JAX objects are immutable is that jax is compiled from it's more friendly numpy api interface (i.e. `import jax.numpy as jnp`) to `lax` to `xla` which stands for Accelerated Linear Algebra. As a result, we can't mutate objects. So then how do we assign values to objects? Lets see:

```
import jax.numpy as jnp
import numpy as np

base = np.array([1, 2, 3, 4, 5])
base[0] = 100
print(base)
# >>> [100, 2, 3, 4, 5]

base_jax = jnp.array([1, 2, 3, 4, 5])
base_jax[0] = 100
print(base_jax)
# >>> TypeError: JAX arrays are immutable

```

In order to make this work, we need to select the index and set the value we want at the given index. Like so:

```
import jax.numpy as jnp
import numpy as np

base = np.array([1, 2, 3, 4, 5])
base[0] = 100
print(base)
# >>> [100, 2, 3, 4, 5]

base_jax = jnp.array([1, 2, 3, 4, 5])
# Notice the square brackets
# and the assignment to the variable updated_base (functional programming:))
updated_base = base_jax.at[0].set(100)
print(updated_base)
# >>> [100, 2, 3, 4, 5]
```

So far so good! Now lets speed things up. To the moon!!! (I'm so happy I found a way to include a crypto joke)

# Jit: Speed it up!
Jit stands for Just in Time compilation. Although how the name JAX came to be remains unclear, some argue the  J in JAX stands for JIT, or Just-In-Time. Jit compiles your code to XLA so that it can efficiently be executed on the compute you use (e.g. TPUs).

Without jit compilation, your code would run okay-ish.
```
import jax
import jax.numpy as jnp

def selu(x, alpha=1.67, lambda_=1.05):
  return lambda_ * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

x = jnp.arange(1000000)
%timeit selu(x).block_until_ready()
>>> 1.64 ms ± 32.5 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

```
We see that each run takes on average 1.64 milliseconds and has a standard devation of 32.5 microseconds. If you don't know what a standard deviation is, no worries. Just have a look at the average for now.

Now lets jit the function. that is, let's compile!
```
selu_jit = jax.jit(selu)

# Warm up
selu_jit(x).block_until_ready()

%timeit selu_jit(x).block_until_ready()
>>> 401 µs ± 26 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
```

Our unjitted version was roughly %300 percent slower. Not bad to achieve such a speed up so easily!

# Vmap:
You might be familiar with map and apply in python. Well, that's basically Vmap or Vectorizedmap in JAX. Ofcourse, plenty of stuff is actually happing, but this might make it easy to get started with. Don't worry too much about the details for now. You can worry about that once you feel confident working with DL libraries and JAX in general.

So what `vmap` does, it takes your code, or your function, and maps it over the input parameters. Now why do we need this in the first place? The following example shows that we would need to write different code based on a batch or single sample.

```
from jax import jit, vmap
from jax import random 

keygen = random.PRNGKey(0)
weights = random.normal(keygen, shape=(20, 2)) # hidden layer with 20 neurons and 2 inputs
features = random.normal(keygen, shape=(5, 2)) # batch of 5 samples with two features

def dotproduct(w, x):
    return jnp.dot(w, x) 

# dotproduct works on a single sample, but we want to apply it to all samples in the batch.
dotproduct(10, 20) #works :)

# now we want to apply it to all samples in the batch
dotproduct = jit(dotproduct)
dotproduct(weights, features)

# >>> TypeError: Incompatible shapes for dot: got (20, 2) and (5, 2).


```

With vmap, we can easily write our code for a single sample and than map every sample to that single function, like so:

```

@jit
def linear_forward(weights, features):
    batch_dim_weights = None
    batch_dim_features = 0
    return vmap(dotproduct, in_axes=(batch_dim_weights, batch_dim_features))(weights, features)



preds = linear_forward(weights, features)
preds.shape

>> (5, 20)

```

Great! Easy as you like. 


## Conclusion
JAX is the new kid on the block providing some powerful capabilities. When you get started, you might need to wrap your head around some of the basics, but once you've got that covered, its a joy to work with. Want to run the examples yourself, check out this notebook: [](). Still questioning whether JAX is something for you, checkout my other blogpost: [](). Ready to train your first machine learning model with jax.numpy, jax.grad, checkout my other notebook: []()




