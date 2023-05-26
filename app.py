import requests
from io import BytesIO
import base64
from PIL import Image
from nsfw_model import NsfwModel

"""
url : required, image url
b64: base64-png string
prd/stg/dev
"""

nsfw = NsfwModel()
# nsfw = NsfwModel('./open_nsfw_weights_v_1.0.0.h5')

def read_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert('RGB')

def read_from_b64(b64_str):
    pil_img = Image.open(BytesIO(base64.b64decode(b64_str)))
    return pil_img

def lambda_handler(event, context):
    # warmup 용 요청일 경우
    if event.get('warmup'):
        print("warmup success")
        return 'warm up'

    elif event.get('url'):
        pil_image = read_from_url(event['url'])

    elif event.get('imageData'):  # b64
        pil_image = read_from_b64(event['imageData'])

    else:
        raise ValueError("No Valid Parameter Offered.")

    result = nsfw.inference(pil_image)

    return result

# if __name__ == '__main__':
#     eve = {
#   "url": "http://cdn.oround.com/artwork/2023/3/13/44032/PD/N4cJM9-20230313233744973.png"
# }
#     print(lambda_handler(event=eve, context=None))