# import torch
# from diffusers import AmusedPipeline
import requests
from dotenv import load_dotenv
import os
import streamlit as st
from io import BytesIO
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_google_vertexai.vision_models import VertexAIImageGeneratorChat
from langchain_community.llms import OpenAI
from langchain.chains import LLMChain
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.prompts import PromptTemplate
import asyncio
from pathlib import Path
import aiohttp
from loguru import logger
from PIL import Image
from horde_sdk import ANON_API_KEY, RequestErrorResponse
from horde_sdk.ai_horde_api.ai_horde_clients import AIHordeAPIAsyncClientSession, AIHordeAPIAsyncSimpleClient
from horde_sdk.ai_horde_api.apimodels import ImageGenerateAsyncRequest, ImageGenerateStatusResponse, ImageGenerationInputPayload, TIPayloadEntry
from horde_sdk.ai_horde_api.fields import JobID

load_dotenv()
def generate_image_dalle(command, apikey):

        os.environ["OPENAI_API_KEY"] = apikey

        llm = OpenAI(temperature=0.5)
        prompt = PromptTemplate(
            input_variables=["image_desc"],
            template="Generate a small prompt to generate an image based on the following description: {image_desc}",
        )
        chain = LLMChain(llm=llm, prompt=prompt)

        image_url = DallEAPIWrapper().run(chain.run(command))

        return image_url

# def generate_image_amused(prompt):
#         pipe = AmusedPipeline.from_pretrained(
#             "amused/amused-512", variant="fp32", torch_dtype=torch.float16
#         )
#         # pipe = pipe.to("cuda")
#
#         negative_prompt = "low quality, ugly"
#
#         image = pipe(prompt, negative_prompt=negative_prompt, generator=torch.manual_seed(0)).images[0]
#         return image #TODO: need to change formatting of image to display


# def generate_image_google(command):
#         generator = VertexAIImageGeneratorChat()
#         messages = [HumanMessage(content=[command])]
#         response = generator.invoke(messages)
#         image = response.content[0]

def generate_image_lime(prompt, apikey):
        url = "https://api.limewire.com/api/image/generation"

        payload = {
            "prompt": prompt,
            "aspect_ratio": "1:1"
        }

        headers = {
            "Content-Type": "application/json",
            "X-Api-Version": "v1",
            "Accept": "application/json",
            "Authorization": "Bearer " + apikey
        }

        response = requests.post(url, json=payload, headers=headers)

        data = response.json()
        image_url = data['data'][0]['asset_url']
        return image_url
def get_image_download_link(img_path: str, filename="generated_image.png"):
    if img_path.startswith("http"):
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(img_path)

    buf = BytesIO()
    image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return st.download_button(label="Download Image", data=byte_im, file_name=filename, mime="image/png")


async def generate_image_with_horde(prompt: str, model: str, apikey: str = ANON_API_KEY):
    async def async_image_generate(simple_client: AIHordeAPIAsyncSimpleClient):
        single_generation_response: ImageGenerateStatusResponse
        job_id: JobID

        single_generation_response, job_id = await simple_client.image_generate_request(
            ImageGenerateAsyncRequest(
                apikey=apikey,
                prompt=prompt,
                models=[model],
                params=ImageGenerationInputPayload(
                    height=1024,
                    width=1024,
                    tis=[
                        TIPayloadEntry(
                            name="72437",
                            inject_ti="negprompt",
                            strength=1,
                        ),
                    ],
                    n=1,
                ),
            ),
        )

        if isinstance(single_generation_response, RequestErrorResponse):
            logger.error(f"Error: {single_generation_response.message}")
            return None, single_generation_response.message
        else:
            example_path = Path("requested_images")
            example_path.mkdir(exist_ok=True, parents=True)

            download_image_tasks: list[asyncio.Task[tuple[Image, JobID]]] = []

            for generation in single_generation_response.generations:
                download_image_tasks.append(asyncio.create_task(simple_client.download_image_from_generation(generation)))

            downloaded_images: list[tuple[Image, JobID]] = await asyncio.gather(*download_image_tasks)

            for image, job_id in downloaded_images:
                filename_base = f"{job_id}_simple_async_example"
                image_path = example_path / f"{filename_base}.webp"
                image.save(image_path)
                logger.info(f"Image saved to {image_path}")

                return str(image_path), None

    aiohttp_session = aiohttp.ClientSession()
    horde_client_session = AIHordeAPIAsyncClientSession(aiohttp_session)

    async with aiohttp_session, horde_client_session:
        simple_client = AIHordeAPIAsyncSimpleClient(
            aiohttp_session=aiohttp_session,
            horde_client_session=horde_client_session,
        )

        image_path, error = await async_image_generate(simple_client)
        return image_path, error

def generate_image_with_horde_sync(prompt: str, model: str, apikey: str = ANON_API_KEY):
    return asyncio.run(generate_image_with_horde(prompt, model, apikey))