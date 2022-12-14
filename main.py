from enum import Enum
import tensorflow.keras as keras
import librosa
import numpy as np
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
import psycopg2.extras
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# engine = create_engine('postgresql://postgres:postgres@runcloud.c2pbcdhtmldg.us-east-1.rds.amazonaws.com:5432/RunCloud')
# create table covid_detection(id SERIAL PRIMARY KEY, age INTEGER, is_return_user BOOLEAN, result VARCHAR(255), cough_heavy_file_location VARCHAR(255));
# engine.execute("CREATE TABLE IF NOT EXISTS covid_detection (id SERIAL PRIMARY KEY, age INTEGER, is_return_user BOOLEAN, result VARCHAR(255), cough_heavy_file_location VARCHAR(255))")

conn = psycopg2.connect(
    host="localhost",
    database="covid-detection",
    user="postgres",
    password="postgres"
)

model = keras.models.load_model('./models/cough_model.h5')
@app.post("/covid_detection")
async def create_item(age: str = Form(), is_return_user: bool = Form(), coughBlob: object = Form()):
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("INSERT INTO covid_detection (age, is_return_user, result, cough_heavy_file_location) VALUES (%s, %s, %s, %s) RETURNING id;", (age, is_return_user, "pending", "/cough/"+coughBlob.filename))
        id_of_new_row = cur.fetchone()[0]
        with open("./cough.wav", "wb+") as file_object:
            file_object.write(coughBlob.file.read())
        
        os.mkdir("./data/"+str(id_of_new_row))

        with open("./data/"+ str(id_of_new_row) + "/" + coughBlob.filename, "wb+") as file_object:
            file_object.write(coughBlob.file.read())

        covid_status_name = ['healthy', 'no respillness exposed', 'resp illness not identified', 'positive moderate', 'recovered full', 'positive mild', 'positive asymp', 'under validation']
        y,sr = librosa.load("./cough.wav")
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128, fmax=8000)
        S_dB_mel = librosa.amplitude_to_db(S, ref=np.max)
        S_dB_mel = S_dB_mel[:128, :225]
        S_dB_mel_np_arr = np.array([S_dB_mel])
        p = model.predict(S_dB_mel_np_arr)
        cur.execute("UPDATE covid_detection SET result = %s WHERE id = %s", (covid_status_name[np.argmax(p)], id_of_new_row))
        print(p)
        print(p[0][np.argmax(p)])
        p = np.argmax(p, axis=1)
        print(p)

        return {"age": age, "is_return_user": is_return_user, "cough": coughBlob.filename, "covid_status": covid_status_name[p.tolist()[0]]}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/covid_detection")
async def get_all_items():
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
        cur.execute("SELECT * FROM covid_detection")
        rows = cur.fetchall()
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))