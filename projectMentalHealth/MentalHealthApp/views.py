from django.shortcuts import render
from joblib import load
import numpy as np
from . import *

# Create your views here.


def predictor(request):
    if request.method == "POST":

        # Extract form data
        depression_score = float(request.POST.get("dep_s"))
        anxiety_score = float(request.POST.get("anx_s"))
        productivity_score = float(request.POST.get("pro_s"))

        # Load model and predict
        model = load(
            "D:\\XCode\\Work\\python\\Projects\\MentalHealth\\model\\model.joblib"
        )

        features = np.array(
            [[depression_score, anxiety_score, productivity_score]]
        )
        pred = model.predict(features)[0]

        print("----------------------------------------------------Prediction:", (pred))
        context = {
            "depression_score": depression_score,
            "anxiety_score": anxiety_score,
            "productivity_score": productivity_score,
            "risk_score": pred.astype(int),

        }

        return render(request, "result.html", {"result": context})
    return render(request, "index.html")
