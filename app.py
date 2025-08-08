from flask import Flask, request, render_template
import pandas as pd
import pickle
import resources.data as data

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
@app.route('/first') 
def first():
	return render_template('first.html')

@app.route('/upload') 
def upload():
	return render_template('upload.html') 
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)   
@app.route('/login') 
def login():
	return render_template('login.html') 
@app.route('/chart') 
def chart():
	return render_template('chart.html') 

@app.route('/performance') 
def performance():
	return render_template('performance.html')     


@app.route('/home')
def home():
    return render_template('index.html',states = data.state, junctions = data.junction, vechicleAge = data.vehicle_age, 
                           humanAgeSex = data.human_age_sex, personWithoutPrecautions = data.person_without_precautions, 
                           areas = data.area, typeOfPlace = data.type_of_place, vehicleLoad = data.vehicle_load, 
                           trafficRulesViolation = data.traffic_rules_violation, weather = data.weather, 
                           vehicleTypeSex = data.vehicle_type_sex, roadType = data.road_type, License = data.license_type, 
                           time = data.time)


@app.route('/predict', methods = ['POST'])
def predict():
    userFeatures = [x for x in request.form.values()]
    
    testData = {'States/UTs':[userFeatures[0]], 'JUNCTION':[userFeatures[1]], 'VEHICLE AGE':[userFeatures[2]],
                'HUMAN AGE AND SEX':[userFeatures[3]], 'PERSON WITHOUT SAFETY PRECAUTIONS':[userFeatures[4]],
                'AREA':[userFeatures[5]], 'TYPE OF PLACE':[userFeatures[6]], 'LOAD OF VEHICLE':[userFeatures[7]],
                'TRAFFIC RULES VIOLATION':[userFeatures[8]], 'WEATHER':[userFeatures[9]], 
                'VEHICLE TYPE AND SEX':[userFeatures[10]], 'TYPE OF ROAD':[userFeatures[11]], 'LICENSE':[userFeatures[12]],
                'TIME':[userFeatures[13]]}
    
    for col in testData:
        code = [data.columnCodes[''.join(testData[col])]]
        testData[col] = code
    
    testDataFrame = pd.DataFrame.from_dict(testData)
    
    prediction = model.predict(testDataFrame)
 
    if prediction[0] == 0:
        output = "No, There is No Chance of Road Accident."
    else:
        output = "Yes, There is a Chance Of Road Accident! Be Careful."
    
    return render_template('index.html',states = data.state, junctions = data.junction, vechicleAge = data.vehicle_age, 
                           humanAgeSex = data.human_age_sex, personWithoutPrecautions = data.person_without_precautions, 
                           areas = data.area, typeOfPlace = data.type_of_place, vehicleLoad = data.vehicle_load, 
                           trafficRulesViolation = data.traffic_rules_violation, weather = data.weather, 
                           vehicleTypeSex = data.vehicle_type_sex, roadType = data.road_type, License = data.license_type, 
                           time = data.time,prediction_text = output)  

if __name__ == "__main__":
    app.run()
