from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
import datetime
import openpyxl

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import VotingClassifier

# Create your views here.
from Remote_User.models import ClientRegister_Model,Popularity_prediction,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city)

        return render(request, 'RUser/Register1.html')
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Predict_Popularity_Type(request):
    if request.method == "POST":
        tmessage = request.POST.get('keyword')

        df = pd.read_csv('Tweets.csv')
        df
        df.columns
        df.isnull().sum()
        df.rename(columns={'likes': 'Likes', 'tweet': 'Tweet'}, inplace=True)

        def apply_recommend(Likes):
            if (Likes <= 1000):
                return 0  # Less Popularity
            elif (Likes <= 5000 and Likes >= 1000):
                return 1  # Average Popularity
            elif (Likes <= 100000 and Likes >= 5000):
                return 2  # Popularity
            elif (Likes <= 500000 and Likes >= 100000):
                return 3  # More Popularity

        df['Popularity'] = df['Likes'].apply(apply_recommend)
        df.drop(['Likes'], axis=1, inplace=True)
        Popularity = df['Popularity'].value_counts()
        df.drop(['id', 'timestamp', 'url', 'replies', 'retweets', 'quotes'], axis=1, inplace=True)

        cv = CountVectorizer()
        X = df['Tweet']
        y = df['Popularity']

        print("Tweet")
        print(X)
        print("Popularity")
        print(y)

        X = cv.fit_transform(X)

        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("Naive Bayes")

        from sklearn.naive_bayes import MultinomialNB
        NB = MultinomialNB()
        NB.fit(X_train, y_train)
        predict_nb = NB.predict(X_test)
        naivebayes = accuracy_score(y_test, predict_nb) * 100
        print(naivebayes)
        print(confusion_matrix(y_test, predict_nb))
        print(classification_report(y_test, predict_nb))
        models.append(('naive_bayes', NB))

        # SVM Model
        print("SVM")
        from sklearn import svm
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))

        print("Logistic Regression")

        from sklearn.linear_model import LogisticRegression
        reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('logistic', reg))

        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        tweet_data = [tmessage]
        vector1 = cv.transform(tweet_data).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if prediction == 0:
            val = 'Less Popularity'
        elif prediction == 1:
            val = 'Average Popularity'
        elif prediction == 2:
            val = 'Above Average Popularity'
        elif prediction == 3:
            val = 'More Popularity'

        print(val)
        print(pred1)

        predicts = 'predicts.csv'
        df.to_csv(predicts, index=False)
        df.to_markdown

        print(val)
        print(pred1)

        Popularity_prediction.objects.create(Tweet_Message=tmessage,Prediction=val)

        return render(request, 'RUser/Predict_Popularity_Type.html',{'objs': val})
    return render(request, 'RUser/Predict_Popularity_Type.html')



