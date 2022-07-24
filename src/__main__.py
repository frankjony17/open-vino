import os

import uvicorn
from fastapi import FastAPI
from starlette.staticfiles import StaticFiles

from src.controller import open_video_controller

static_path = os.path.dirname(os.path.abspath(__file__ + '/..'))

app = FastAPI(
    title='OpenVINO - API',
    description='Boost deep learning performance in computer vision, automatic speech recognition, '
                'natural language processing and other common tasks.')

app.mount("/static", StaticFiles(directory=static_path + '/static/'), name="static")

app.include_router(open_video_controller.router, prefix="/openvino", tags=["OpenVINO"])

if __name__ == "__main__":
    uvicorn.run("__main__:app", host="0.0.0.0", log_level="debug")
