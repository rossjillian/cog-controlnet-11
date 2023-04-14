# Cog implementation of ControlNet 1.1

This is an implementation of [ControlNet 1.1](https://github.com/lllyasviel/ControlNet-v1-1-nightly) as a Cog model. [Cog](https://github.com/replicate/cog) packages machine learning models as standard containers.

First, download the pre-trained weights.

Then, you can run predictions:

`cog predict -i image=@test.png -i prompt="test" -i structure="scribble"`
