# Grokking JAX: How to get started with Deep Learning in Jax

If you have been developing your own deep learning models, you are probably familiar with the most popular deep learning frameworks out there: PyTorch and Tensorflow. However there is a new kid on the block and that is JAX. If you are still deciding whether JAX is right for you, check out my other blogpost _[5 Reasons why you should choose JAX for your deep learning projects instead of PyTorch or Tensorflow]()_.

## JAX a.k.a. numpy on steriods

So you have decided to explore JAX to see if it is something for you. Great! You open your browser, go to the official JAX documentation to better understand what JAX is. You are probably expecting something in the lines of:

_JAX: An open source machine learning framework that accelerates the path from research prototyping to production deployment._

But when you land on the homepage of JAX, well read for yourself...

_JAX is autograd and XLA brought together for high-performance numerical computing and machine learning research. It provides composable transformations of Python+NumPy programs: differentiate, vectorize, parallelize, Just-In-Time compile to GPU/TPU._

That's quite a lot to take in all in once. So lets dissect it. Essentially it is saying, JAX take care of computating the gradients for me so that I can train my machine learning Model. Ooh, and while we are at it, my models are quite big so make sure that I can ran them on any compute I want, whether that's CPUs, GPUs or even TPUs in a highly efficient way. That sounds better! So now what? Teach me the basics. Or better yet, show me some code!

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
# and the assignment to the variable updated_base
# we'll see why we need the assignment a bit later
updated_base = base_jax.at[0].set(100)
print(updated_base)
# >>> [100, 2, 3, 4, 5]
```

So far so good! Now lets speed things up. To the moon!!! (so happy I found a way to include a crypto joke in here)

# Jit: Speed it up!

# Vmap:

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
