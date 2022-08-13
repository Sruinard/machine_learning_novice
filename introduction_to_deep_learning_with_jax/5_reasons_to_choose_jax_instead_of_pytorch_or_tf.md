# 5 Reasons why you should choose JAX for your deep learning projects instead of PyTorch or Tensorflow

Tensorflow and PyTorch are two of the most widely used deep learning frameworks out there. PyTorch, coming from Meta, started out as more oriented towards researchers, whereas Tensorflow, backed by Google, was generally more accepted within the industry, but even that has arguably changed in recent times. What can be stated which quite some confidence is that these two giants have put many of the other once popular frameworks to rest (CNTK, caffe, etc). However, a surprising event has happened with JAX entering the field. JAX is the latest arrival and is likely to shake things up quite a bit! So here are 5 reasons why you should JAX for your next deep learning project.

# Backed by Google Brain and Google DeepMind
The industry standard might arguably still be Tensorflow, but PyTorch has taken quite a big chunk out of Tensorflow's marketshare. Discussions on reddit even mention that teams within Google are becoming more hesitant to be using Tensorflow due to it's inconsistent API and that teams are moving towards JAX. Don't believe change is happening? DeepMind, probably one of the most advanced AI companies out there, has standardized on JAX. They build their own framework on top of it called Haiku. In addition, Google Brain started a similar endeavour by creating Flax. So the company who created Tensorflow in the first place is putting more chips behind JAX.

# 5000+ models available on HuggingFace
It's one thing that these big tech companies invest in new technologies. They have plenty of resources to make small bets and see how it will turn out. Sure, but don't expect this to be just a small experiment. There are already over 5000 JAX models hosted on HuggingFace and they even provide JAX as an interface to simplify submitting your models. Why is that important? As DeepMind and Google are pushing the AI frontier further, they'll be increasingly releasing SOTA models in JAX to HuggingFace rather than in Tensorflow. Consequenlty, we'll see more benefits of using JAX because those who can incorporate JAX in their enterprise will gain a competitive edge.

# Easily convert JAX to Tensorflow
But JAX is less mature than Tensorflow and PyTorch you might say. They have a massive ecosystem and mature production-ready features to bring the value from these deep learning models to our organizations, while JAX is still lacking in that regard, right? Well... maybe... JAX provides quite some powerful features. One if which is `from jax.experimental import jax2tf #(finally, a code block ;))`. This enables you to develop your models in JAX, but leverage industry tools such as Tensorflow Serving for model deployment. Not bad!

# Run on CPUs/GPUs/TPUs


# Familiar Numpy-like API

# Conclusion