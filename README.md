A simple (um... no) example of a discord bot which runs huggingface diffusers' controlnet pipeline (the canny edges one, to be specific) in response to a slash command with an input image attached, followed by a prompt.

It should serve as a decent starting point for anyone who wants to implement a discord-bot interface to huggingface's diffusers pipelines.

There is a lot left to do here, but the basics are working, the bot will respond with a message contaiing the intermediate results of the diffusion, and update the image periodically until it gets the final result from the pipeline.

This was no small feat, as discord.py uses asyncio and diffusers' pipelines are cpu bound and the progress callback isv not async friendly either. we end up needing to run the pipeline inside a thread and use a janus.Queue to send the intermediate and final images back to the async main thread so it can edit the bot's message.

This had me stumped for a good 24 hours or so, and I'm really grateful for janus which made everything work in about 10 minutes once I came across it. Shout out to that project's maintainers.
