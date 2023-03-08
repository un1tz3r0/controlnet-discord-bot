# controlnet-discord-bot

A simple (not really tho) example of a discord bot which runs huggingface [diffusers](https://huggingface.co/docs/diffusers/index) [controlnet pipeline](https://github.com/lllyasviel/ControlNet) (the canny edges one, to be specific) in response to a slash command (`/cannyremix <prompt>`)with an input image attachment, and a prompt after the slash-command.

It is by no means complete, but should serve as a decent starting point for anyone who wants to implement a discord-bot interface to huggingface's diffusers pipelines.

## What is controlnet?

Huggingface has a great [blog post](https://huggingface.co/blog/controlnet) where you can read more about all the crazy stuff you can do with controlnet.

# Getting Started

Install controlnet dependencies. This should grab torch and everything else you need (I think, feel free to open an issue if I missed something)

```
pip install diffusers==0.14.0 transformers xformers git+https://github.com/huggingface/accelerate.git

pip install opencv-contrib-python
pip install controlnet_aux
```

Install janus Queue

```
pip install git+https://github.com/aio-libs/janus
```

Install [py-cord](https://docs.pycord.dev/en/stable/installing.html)...

```
python3 -m pip install -U py-cord --pre
```

# Future

There is a lot left to do here, but the basics are working, the bot will respond with a message containing the intermediate results of the diffusion, and update the image periodically until it gets the final result from the pipeline.

# Comments

This was no small feat, as discord.py uses asyncio and diffusers' pipelines are cpu bound and the progress callback isv not async friendly either. we end up needing to run the pipeline inside a thread and use a [janus](https://github.com/aio-libs/janus).Queue to send the intermediate and final images back to the async main thread so it can edit the bot's message.

This had me stumped for a good 24 hours or so, and I'm really grateful for [janus]((https://github.com/aio-libs/janus) which made everything work in about 10 minutes once I came across it.