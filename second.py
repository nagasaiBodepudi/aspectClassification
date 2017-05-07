#This is a python 2.7 code
#Required Library's
#Clustering is done using KMeans algorithm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import nltk

def read_file():
	#opening the file
	with open("aspect_words_1k_restaurant_reviews","rt") as in_file:
		data = in_file.read();
	return data;

def tokenizer(data):
	#Dividing the file into tokens
	tokens = nltk.word_tokenize(data);
	#print(tokens)  # debug statement
	return tokens;

def tfidvectorizer(tokens):
	vectorizer = TfidfVectorizer(stop_words='english')
	X = vectorizer.fit_transform(tokens)
	#print(X)
	print(X.shape)
	return X;

def clustering(tfid_matrix, num_clusters):
	km = KMeans(n_clusters = num_clusters);
	km.fit(tfid_matrix);
	return km;


def main():
	data = read_file();
	tokens = tokenizer(data);
	tfid_matrix = tfidvectorizer(tokens);
        n = len(tokens);
	#print(n);
	num_clusters = n/20; #suppose 1000 words are given then they are divided in to 50 clusters
	#print(num_clusters);
	km = clustering(tfid_matrix, num_clusters);
	order_centroids = km.cluster_centers_.argsort()[:, ::-1];#ordering
	for i in range(num_clusters):
		cluster = [];
		for ind in order_centroids[i, :]:
    			cluster = cluster + [tokens[ind]];
		#	print(tokens[ind]);
		#	print('**')
		#print('*********')
		print(cluster);
		print;
	

if __name__ == "__main__":
      main()


