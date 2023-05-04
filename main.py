import shutil
import cv2
import tensorflow as tf
import numpy as np
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import File, Request
from fastapi import FastAPI, UploadFile
from starlette.responses import RedirectResponse
import starlette.status as status
from Model import model
from PIL import Image
app = FastAPI()
templates = Jinja2Templates(directory="templates")
@app.get("/", response_class=HTMLResponse)
def upload(request: Request):
   return templates.TemplateResponse("index.html", {"request": request})

def predict():
   img = cv2.imread('destination.png')
   # img = Image.open("destination.png",'r')
   print(type(img))
   img = tf.image.resize(
      img,
      [224, 224]
   )
   img = np.expand_dims(img, axis=0)
   pred = model.predict(img)
   pred = np.squeeze(pred)
   print(pred)
   value = np.where(pred == max(pred))
   print("Accuracy: " + str(pred[value]))
   process_pred = value[0][0]
   print(process_pred)
   D = {0: 'S', 1: 'M', 2: 'Q', 3: 'N', 4: 'F', 5: 'V'}
   print("Predict: " + str(D[process_pred]) + " Accuracy: " + str(pred[value][0]*100)+"%")
   return str(D[process_pred]), str(pred[value][0]*100)

@app.post('/upload/file', response_class=HTMLResponse)
def create_upload_file(file: UploadFile = File(...)):
   with open("destination.png", "wb") as buffer:
      shutil.copyfileobj(file.file, buffer)
      print(type(file))
      pred, acc = predict()
      request = RedirectResponse(f"http://127.0.0.1:8000/result/{pred}/{acc}",  status_code=status.HTTP_302_FOUND)
      return request

@app.get('/result/{pred}/{acc}',response_class=HTMLResponse)
def result(request: Request, pred: str, acc: str):
   return templates.TemplateResponse("result.html", {"request": request, "pred": pred, "acc": acc})
