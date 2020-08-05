from django.shortcuts import render
from .house import predict_price

# Create your views here.
def Index(request):
	return render(request,'index.html')

def Prediction(request):
	if request.method == "POST":
		location = request.POST['location']
		sqft = request.POST['sqft']
		bath = request.POST['bath']
		bhk = request.POST['bhk']


		house = predict_price(location,sqft,bath,bhk)

		return render(request,'prediction.html',{'house':house})
	return render('request','prediction.html')

def Firstpage(request):
	return render(request,'firstpage.html')
























































