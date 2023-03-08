import cv2
from PIL import Image
import numpy as np
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from diffusers import UniPCMultistepScheduler
import threading, asyncio

'''
to install dependencies:

    pip install diffusers==0.14.0 transformers xformers git+https://github.com/huggingface/accelerate.git
    pip install opencv-contrib-python
    pip install controlnet_aux
'''

import asyncio
import queue
import threading
import janus

class AsyncQueue:
    def __init__(self, loop):
        self.regular_queue = queue.Queue()
        self.asyncio_queue = asyncio.Queue()
        self.lock = threading.RLock()
        self.loop = loop
        self.worker_thread = threading.Thread(target=self.queue_worker)
        self.worker_thread.start()

    def put(self, item):
        with self.lock:
            self.regular_queue.put(item)

    async def get(self):
        return await self.asyncio_queue.get()

    def stop(self):
        with self.lock:
            self.regular_queue.put(None)

    def queue_worker(self):
        while True:
            with self.lock:
                if not self.regular_queue.empty():
                    item = self.regular_queue.get()
                    asyncio.run_coroutine_threadsafe(self.asyncio_queue.put(item), self.loop)
                    continue
            asyncio.run_coroutine_threadsafe(self.asyncio_queue.put(None), self.loop)
            break


async def consume_items(queue):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        yield item
        #queue.task_done()


class CannyDiffusionImageTransform:

    async def __call__(self, image, prompt, seed=None, num_inference_steps=20, negative_prompt=None, callback=None):
        ''' convert it to edges '''
        
        image = np.array(image)
        # canny edge detection preprocessor
        low_threshold = 100
        high_threshold = 200
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)
        # allow setting the seed
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = [torch.Generator(device="cpu").manual_seed(seed)]
        
        # run the pipelione inside a thread while waiting on a queue
        # this is so we can update the image in the discord message
        # without blocking the bot

        aq = janus.Queue() #AsyncQueue(asyncio.get_running_loop())

        async def produce_items():
            for i in range(10):
                aq.put(i)
                await asyncio.sleep(0.5)

            aq.stop()


        def _run_pipeline(self, prompt, canny_image, generator, num_inference_steps, negative_prompt):
            def progress_callback(step, total_steps, latents):
                try:
                    # convert latents to image
                    with torch.no_grad():
                        latents = 1 / 0.18215 * latents
                        image = self.pipe.vae.decode(latents).sample

                        image = (image / 2 + 0.5).clamp(0, 1)

                        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
                        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

                        # convert to PIL Images
                        image = self.pipe.numpy_to_pil(image)

                        aq.sync_q.put((step, total_steps, image[0]))
                except Exception as e:
                    print(f"progress_callback exception: {e}")
                    pass
            
            output = self.pipe(
                str(prompt),
                canny_image,
                #negative_prompt=[negative_prompt],
                num_inference_steps=num_inference_steps,
                generator=generator,
                callback=progress_callback, 
                callback_steps=num_inference_steps//4
            )
            aq.sync_q.put((num_inference_steps, num_inference_steps, output.images[0]))
            aq.sync_q.put(None)
        
        print("Starting producer task")
        producer_task = threading.Thread(target=_run_pipeline, args=[self, prompt, canny_image, generator, num_inference_steps, negative_prompt])
        producer_task.start()
        print("Starting consumer loop")

        async for item in consume_items(aq.async_q):
            print(f"Got item {item} from queue")
            if item == None:
                break
            if callback != None:
                await callback(item[0], item[1], item[2])
        
        print("Done consuming items from queue, awaiting producer task")
        producer_task.join()
        print("Done awaiting producer task")

    def __init__(self):
        # setup the pipeline
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16)

        # make it faster
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        pipe.enable_xformers_memory_efficient_attention()

        self.pipe = pipe
