
from django.db.models import  Count, Avg
from django.shortcuts import render, redirect
from django.db.models import Count
from django.db.models import Q
import datetime
import xlwt
from django.http import HttpResponse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.ensemble import VotingClassifier

import warnings
warnings.filterwarnings("ignore")
plt.style.use('ggplot')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# Create your views here.
from Remote_User.models import ClientRegister_Model,Popularity_prediction,detection_ratio,detection_accuracy


def serviceproviderlogin(request):
    if request.method  == "POST":
        admin = request.POST.get('username')
        password = request.POST.get('password')
        if admin == "Admin" and password =="Admin":
            return redirect('View_Remote_Users')

    return render(request,'SProvider/serviceproviderlogin.html')

def Find_Popularity_Type_Ratio(request):
    detection_ratio.objects.all().delete()
    ratio = ""
    kword = 'Less Popularity'
    print(kword)
    obj = Popularity_prediction.objects.all().filter(Q(Prediction=kword))
    obj1 = Popularity_prediction.objects.all()
    count = obj.count();
    count1 = obj1.count();
    ratio = (count / count1) * 100
    if ratio != 0:
        detection_ratio.objects.create(names=kword, ratio=ratio)

    ratio1 = ""
    kword1 = 'Average Popularity'
    print(kword1)
    obj1 = Popularity_prediction.objects.all().filter(Q(Prediction=kword1))
    obj11 = Popularity_prediction.objects.all()
    count1 = obj1.count();
    count11 = obj11.count();
    ratio1 = (count1 / count11) * 100
    if ratio1 != 0:
        detection_ratio.objects.create(names=kword1, ratio=ratio1)

    ratio12 = ""
    kword12 = 'Above Average Popularity'
    print(kword12)
    obj12 = Popularity_prediction.objects.all().filter(Q(Prediction=kword12))
    obj112 = Popularity_prediction.objects.all()
    count12 = obj12.count();
    count112 = obj112.count();
    ratio12 = (count12 / count112) * 100
    if ratio12 != 0:
        detection_ratio.objects.create(names=kword12, ratio=ratio12)

    ratio122 = ""
    kword122 = 'More Popularity'
    print(kword122)
    obj122 = Popularity_prediction.objects.all().filter(Q(Prediction=kword12))
    obj1122 = Popularity_prediction.objects.all()
    count122 = obj122.count();
    count1122 = obj1122.count();
    ratio122 = (count122 / count1122) * 100
    if ratio122 != 0:
        detection_ratio.objects.create(names=kword122, ratio=ratio122)

    obj = detection_ratio.objects.all()
    return render(request, 'SProvider/Find_Popularity_Type_Ratio.html', {'objs': obj})

def View_Remote_Users(request):
    obj=ClientRegister_Model.objects.all()
    return render(request,'SProvider/View_Remote_Users.html',{'objects':obj})

def ViewTrendings(request):
    topic = Popularity_prediction.objects.values('topics').annotate(dcount=Count('topics')).order_by('-dcount')
    return  render(request,'SProvider/ViewTrendings.html',{'objects':topic})

def charts(request,chart_type):
    chart1 = detection_ratio.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts.html", {'form':chart1, 'chart_type':chart_type})

def charts1(request,chart_type):
    chart1 = detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/charts1.html", {'form':chart1, 'chart_type':chart_type})

def View_Popularity_Predicted_Type(request):
    obj =Popularity_prediction.objects.all()
    return render(request, 'SProvider/View_Popularity_Predicted_Type.html', {'list_objects': obj})

def likeschart(request,like_chart):
    charts =detection_accuracy.objects.values('names').annotate(dcount=Avg('ratio'))
    return render(request,"SProvider/likeschart.html", {'form':charts, 'like_chart':like_chart})


def Download_Trained_DataSets(request):

    response = HttpResponse(content_type='application/ms-excel')
    # decide file name
    response['Content-Disposition'] = 'attachment; filename="TrainedData.xls"'
    # creating workbook
    wb = xlwt.Workbook(encoding='utf-8')
    # adding sheet
    ws = wb.add_sheet("sheet1")
    # Sheet header, first row
    row_num = 0
    font_style = xlwt.XFStyle()
    # headers are bold
    font_style.font.bold = True
    # writer = csv.writer(response)
    obj = Popularity_prediction.objects.all()
    data = obj  # dummy method to fetch data.
    for my_row in data:
        row_num = row_num + 1

        ws.write(row_num, 0, my_row.Tweet_Message, font_style)
        ws.write(row_num, 1, my_row.Prediction, font_style)

    wb.save(response)
    return response

def Train_Test_DataSets(request):
    detection_accuracy.objects.all().delete()
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
            return 2  # Above Average Popularity
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
    detection_accuracy.objects.create(names="Naive Bayes", ratio=naivebayes)
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
    detection_accuracy.objects.create(names="SVM", ratio=svm_acc)
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
    detection_accuracy.objects.create(names="Logistic Regression", ratio=accuracy_score(y_test, y_pred) * 100)

    classifier = VotingClassifier(models)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    Tweet = "Day of shame in Congress. Protections for pre-existing conditions, mental health, maternity care, addiction services -- all gone."
    tweet_data = [Tweet]
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

    obj = detection_accuracy.objects.all()


    return render(request,'SProvider/Train_Test_DataSets.html', {'objs': obj})