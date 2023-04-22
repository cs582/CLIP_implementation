import concurrent.futures
import asyncio
import aiohttp
from urllib.request import urlopen, Request
from PIL import Image
import os
import io


async def download_image(session, url, path, name):
    try:
        async with session.get(url) as response:
            content = await response.read()
            with Image.open(io.BytesIO(content)) as image:
                filepath = os.path.join(path, f"{name}.jpg")
                image.save(filepath)
                return f"{name}.jpg"
    except:
        print(f"Error while downloading image {url}.")


def url_image_save_async(urls, path, num_workers=10, first_index=0):
    img_index = first_index
    os.makedirs(path, exist_ok=True)
    async with aiohttp.ClientSession() as session:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            loop = asyncio.get_event_loop()
            tasks = []
            for url in urls:
                name = f"{img_index}"
                task = loop.run_in_executor(executor, download_image, session, url, path, name)
                tasks.append(task)
                img_index += 1
            results = await asyncio.gather(*tasks)
            return results
