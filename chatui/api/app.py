import os
import httpx
import shutil

from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse

import api
from model import Message, MessageTurbo

app = FastAPI()

BASE_URL = os.getenv("ENV_BASE_URL", "")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(BASE_DIR, 'dist')
ASSETS_DIR = os.path.join(DIST_DIR, 'assets')
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")
templates = Jinja2Templates(directory=DIST_DIR)


async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        return JSONResponse(content={"code": 500, "error": {"message": f"{type(exc)} {exc}"}})


app.middleware('http')(catch_exceptions_middleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/", response_class=HTMLResponse)
async def root():
    return templates.TemplateResponse("index.html", {"request": {}, "base_url":
                                                     BASE_URL})


@app.post("/completions")
async def completions(request: Request, message: Message):
    api_key = request.headers.get('api_key')
    res = await api.completions(message, api_key=api_key)
    return res


@app.post("/completions_turbo")
async def completions(request: Request, message: MessageTurbo):
    api_key = request.headers.get('api_key')
    print(message.dict(exclude_none=True))
    return StreamingResponse(
        api.completions_turbo(message, api_key=api_key),
        media_type="text/event-stream",
        background=None
    )

@app.get("/credit_summary")
async def credit_summary(request: Request):
    api_key = request.headers.get('api_key')
    res = await api.credit_summary(api_key=api_key)
    return res

@app.post("/upload")
async def upload(request: Request, image: UploadFile = File(...)):
    try:
        # 定义保存图片的路径
        file_path = f"assets/{image.filename}"

        # 将上传的图片保存到指定路径
        with open(f"{ASSETS_DIR}/{image.filename}", "wb") as f:
            shutil.copyfileobj(image.file, f)

        # 返回图片的本地路径
        return {"file_path": str(file_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000)
