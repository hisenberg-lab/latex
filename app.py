from http import HTTPStatus
import numpy as np
import pix2tex
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import codecs

app = FastAPI(
    title="Image to Latex Convert",
    desription="Convert an image of math equation into LaTex code.",
)

@app.on_event("startup")
async def load_model():
    global args
    global objs
    args, *objs = pix2tex.initialize()


@app.get("/", tags=["General"])
def read_root():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.post("/predict/", tags=["Prediction"])
def predict(file: UploadFile=File(...)):
    image = Image.open(file.file)
    prediction = pix2tex.call_model(args, *objs, img=image)
    # replace <, > with \lt, \gt so it won't be interpreted as html code
    prediction = prediction.replace('<', '\\lt ').replace('>', '\\gt ')

    # prediction = pix2tex.output_prediction(prediction, args)
    prediction = codecs.decode(prediction,"unicode_escape")
    prediction = "$$"+prediction+"$$"
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"pred": prediction},
    }
    return response

if __name__ == '__main__':

    uvicorn.run(app, host='127.0.0.1', port=4000, debug=True)