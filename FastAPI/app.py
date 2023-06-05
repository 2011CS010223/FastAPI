
# 1. Install uvicorn and fastapi
# pip install fastapi uvicorn

# 2. Imports Libraries
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle

# 3. Create App
app = FastAPI()

# 4. Configure CORS to access API from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 5. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello World'}

# 6. Run App
if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='0.0.0.0')

# 7a. Using GET Method
@app.get("/predictDiabetes")
def getPredictDiabetes(age: int, hypertension: int, heart_disease: int, bmi:float, HbA1c_level:float, blood_glucose_level:float,gender_Male:int, gender_Other:int):
    model = pickle.load(open("ada_model.pkl", "rb"))
    
    prediction = model.predict([[age,hypertension,heart_disease,bmi,HbA1c_level,blood_glucose_level,gender_Male,gender_Other]])
    
    return {
        'diabetes': prediction[0]
    }
# age	hypertension	heart_disease	bmi	HbA1c_level	blood_glucose_level	gender_Male	gender_Other
# 7b. Using POST Method
# from pydantic import BaseModel

# class house(BaseModel):
#     Area: int
#     BedRooms: int 
#     BathRooms: int 

# @app.post("/predict")
# def predictHousePrice(data: house):
#     rgModel = pickle.load(open("reg.pkl", "rb"))

#     data = data.dict()
#     prediction = rgModel.predict([[data["Area"],data["BedRooms"],data["BathRooms"]]])
    
#     return {
#         'Price': prediction[0]
#     }

#-------------
# 8. Run the API with uvicorn with Reload Option - Auto Run after edit source code
# uvicorn app:app --reload

# 9. Test API from Web Browser
# http://127.0.0.1:8000/predictDiabetes?age=53&hypertension=0&heart_disease=0&bmi=27.32&HbA1c_level=7.0&blood_glucose_level=159.0&gender_Male=0&gender_Other=0
