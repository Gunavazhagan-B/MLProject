from flask import Flask,render_template,request
from src.pipeline.predict_pipeline import CustomData,predictPipeline

app=Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=["GET","POST"])
def predict():
    if request.method=="GET":
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
        pred_df=data.get_data_as_data_frame()
        pred_pipeline=predictPipeline()
        result=pred_pipeline.predict_result(pred_df)
        return render_template('home.html',results=result[0])

if __name__=='__main__':
    app.run(debug=True)



