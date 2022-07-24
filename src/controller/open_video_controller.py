from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import PlainTextResponse, StreamingResponse
from starlette.templating import Jinja2Templates

from src.service.face_detection_service import FaceDetectionService

router = APIRouter()
templates = Jinja2Templates(directory="../templates/")
face_detection = None


@router.get("/start/{in_s}")
def start_0(in_s: int, rq: Request):
    global face_detection
    face_detection = FaceDetectionService(in_s, 1, 1, 1, 1)
    return templates.TemplateResponse("index.html", {"request": rq})


@router.get("/start/{in_s}/{ag}/{hp}/{em}/{lm}")
def start_1(in_s: int, ag: int, hp: int, em: int, lm: int, rq: Request):
    global face_detection
    face_detection = FaceDetectionService(in_s, ag, hp, em, lm)
    return templates.TemplateResponse("index.html", {"request": rq})


@router.get("/stop")
async def stop():
    global face_detection
    if face_detection is not None:
        face_detection.terminate()

    return PlainTextResponse(content='STOP OK')


@router.get('/get_frame')
async def get_frame():
    frame = face_detection.start()
    return StreamingResponse(
        content=frame, media_type='multipart/x-mixed-replace; boundary=temp')
