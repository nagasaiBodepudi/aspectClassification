# Following is python2.7 code
#required Libraries
from sklearn import svm
from nltk.classify import apply_features
import nltk 
import math

#Features of the word 
#I used basic naive bayes algorithm for classification 
#Defining the desired features of the word ensures the accuracy of algorithm
#I defined if the first letter is capital then most probably it is going to be the aspect
def word_features(word):
	features = {};
	features['last_letter'] = word[-1].lower();
	#features['first_letter'] = word[0].lower();
	features['Title_case'] = word.istitle();
	if len(word) > 1: 	
		features['suffix2'] = word[-2].lower();
	return features;


def read_file():
	with open("aspect_annoated_file.txt","rt") as in_file:
		data = in_file.readlines();
	#print(data)
	n = len(data);
	train_n = 3*n/4;
	#print(train_n)
	return data,train_n;

def formatting_data(data,n): #to divide the data into training and testing data set
	total_len = len(data)
	train_data = [];
	test_data = [];
	for i in range(0,total_len):
		tokens = data[i].split();
		#print(tokens); 
		if tokens:
			if i < n:
				if tokens[1] == 'ASP':
					train_data = train_data + [(tokens[0],'ASP')];
				if tokens[1] == 'NASP':
					train_data = train_data + [(tokens[0],'NASP')];
			
			else:
				if tokens[1] == 'ASP':
					test_data = test_data + [(tokens[0],'ASP')];
				if tokens[1] == 'NASP':
					test_data = test_data + [(tokens[0],'NASP')];

	#print('********************************************************************')
	#print(train_data)
	#print('****************************************************************************************************')
	#print(test_data)				
	return train_data,test_data;

	
def ml_algo(train_data,test_data):
        ##This 75% words train and other test##train_n = int(3*n/4); test_n = int(n/4);
	#featuresets = [(word_features(n), aspect) for (n, aspect) in txt];
	train_set = apply_features(word_features, train_data);
        classifier = nltk.NaiveBayesClassifier.train(train_set);
		
	return classifier;
	
def train_analysis(train_data,test_data,classifier):
	errors = [];
	n = len(train_data);	
	error_check = train_data[int(n/4):];
	for (word,tag) in error_check:
		guess = classifier.classify(word_features(word));
		if guess != tag:
			errors.append((tag,guess,word))
	print('Error predictions in trained data:');
	print('Crct'+ ' ' + 'Guess' + ' ' + 'word');	
	for (tag,guess,word) in sorted(errors):
		print(tag+ ' ' + guess + ' ' + word) 

	
def statistics(train_data,test_data,classifier):
	test_set = apply_features(word_features,test_data);
	print('stats:');
	print('probability of Correctness of the Test Data:');	
	print(nltk.classify.accuracy(classifier,test_set));
	tp = 0 #number of true positives
	fp = 0 #number of false positives
	fn = 0 #number of false negatives
	check = [];
	for (word,tag) in test_data:
		guess = classifier.classify(word_features(word));
		check.append((tag,guess,word))
	for (tag,guess,word) in sorted(check):
		if tag == guess :
			tp = tp + 1;
		if tag == 'ASP' and  tag != guess :
			fp = fp + 1;
		if tag == 'NASP' and tag != guess:
			fn = fn + 1;
	print('number of true positive, false positive, flase negative are respectively:')	
	
	print(tp,fp,fn)
	precis = float(tp)/(tp+fp)
	print('Precision:');
	print(precis)
	recall = float(tp)/(tp+fn);
	print('Recall:')
	print(recall)	
	f1 = 2*precis*recall;
	f1 = f1/(precis+recall);
	print('F1-Score:')
	print(f1)


def main():
	data,train_n = read_file();	
	train_data,test_data = formatting_data(data,train_n)
	#print(txt)                      ## Debug statements
	#print(word_features('hello'))
	classifier = ml_algo(train_data,test_data);
	train_analysis(train_data,test_data,classifier);
	statistics(train_data,test_data,classifier); # to check the errors and try to optimize feature function
	#print(classifier.classify(word_features('tamato')))
	print(classifier.show_most_informative_features(5))


if __name__ == "__main__":
      main()


