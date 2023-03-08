import discord
from discord.ext import commands
import PIL.Image, PIL.ImageOps, wand.image, numpy, cv2

from controlnetdemo import CannyDiffusionImageTransform
import curio, asyncio
import PIL, PIL.Image

cannyImageTransform = CannyDiffusionImageTransform()

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='/', intents=intents)

@bot.command()
async def ping(ctx):
		await ctx.send('pong')

@bot.command()
async def echo(ctx, *, content):
		await ctx.send(content)

@bot.command(pass_context=True, )
async def cannyremix(ctx):
		prompt = ctx.message.content.split(' ', 1)[1]
		if len(ctx.message.attachments) > 0:
				# create message to edit later
				outmessage = await ctx.send(f'Processing {len(ctx.message.attachments)} attachments...')
				# download image
				inimages = []
				for attachment in ctx.message.attachments:
						if attachment.content_type.startswith('image/'):
								attachment_data = await attachment.read()
								import PIL.Image, io
								with io.BytesIO(attachment_data) as attachment_stream:
										attachment_image = PIL.Image.open(attachment_stream)
										attachment_image.load()
										inimages.append(attachment_image)
						else:
								await outmessage.edit(content=(outmessage.get_content()  + '\nAttachment is not an image!'))

				outembeds = []
				outattachments = []
				async def add_output_image(outimage, no_edit=False, replace_last=False):
						# cannot replace the last attachment/embed if there aren't any yet
						if replace_last and len(outembeds) == 0:
							replace_last = False
						# turn output image into bytes and upload to discord
						with io.BytesIO() as outstream:
								outimage.save(outstream, 'png')
								outdata = io.BytesIO(outstream.getvalue())
						attachment_num = len(outattachments) - (0 if not replace_last else 1)
						outfile=discord.File(outdata, f'image{attachment_num}.png')
						outembed=discord.Embed()
						outembed.set_image(url=f'attachment://image{attachment_num}.png')
						if not replace_last:
							outembeds.append(outembed)
							outattachments.append(outfile)
						else:
							outembeds[-1] = outembed
							outattachments[-1] = outfile
						if not no_edit:
								# send processed images
								await outmessage.edit(embeds=outembeds, attachments=outattachments)

				# process images
				for inimage in inimages:

						first_image = True
						async def progress_callback(step, total_steps, image):
								nonlocal first_image
								await add_output_image(image, replace_last=not first_image)
								first_image = False
						
						# process image
						await cannyImageTransform(inimage, prompt=prompt, callback=progress_callback)
						
		else:
				await ctx.send('Please attach an image to process!')


@bot.event
async def on_message(message):
		if message.author == bot.user:
				return
		else:
				await bot.process_commands(message)

if __name__=="__main__":
	import dotenv
	dotenv.load_dotenv()
	import os
	if 'DISCORD_TOKEN' in os.environ.keys():
		bot.run(os.environ['DISCORD_TOKEN'])
	else:
		print("DISCORD_TOKEN environment variable not set...")
		print("Please add your DISCORD_TOKEN='...' to a .env file, or use 'env DISCORD_TOKEN=... python bot.py' or add 'setenv DISCORD_TOKEN=...' to your ~/.profile!")
