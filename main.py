import base64
from io import BytesIO

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image

from style_transfer_nn import get_config, prepare_config_to_predict, train

app = FastAPI()

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/")
async def root(request: Request, content_image: UploadFile = File(...)):
    print(content_image.filename)
    return {"ok": "ok"}


@app.post("/upload")
async def upload_img(
    request: Request,
    content_image: UploadFile = File(...),
    style_image: UploadFile = File(...)
):
    content_image_bites = await content_image.read()
    style_image_bites = await style_image.read()
    config = get_config()
    config = prepare_config_to_predict(
        config,
        Image.open(BytesIO(content_image_bites)).convert('RGB'),
        Image.open(BytesIO(style_image_bites)).convert('RGB')
    )
    train(config)
    buffered = BytesIO()
    config["result_images"][-1].save(buffered, format="JPEG")
    return templates.TemplateResponse(
        "result.html", {
            "request": request,
            "image": base64.b64encode(buffered.getvalue()).decode("utf-8")
        }
    )