from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score
import pandas as pd
from sklearn.cross_validation import train_test_split
import random
#from keras.models import Sequential
#from keras.utils import np_utils
#from keras.layers.core import Dense, Activation, Flatten, Dropout
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
import csv
import mca
from sklearn import svm
from sklearn.linear_model import LogisticRegression

def read_test(filename):
	cancer = pd.read_csv(filename)
	clm_list = []
	for column in cancer.columns:
		clm_list.append(column)

	x = cancer[clm_list[3:len(clm_list)]].values
	return x	
	
def read_data(filename):
	cancer = pd.read_csv(filename)
	clm_list = []
	for column in cancer.columns:
		clm_list.append(column)

	x = cancer[clm_list[3:len(clm_list)-1]].values
	y = cancer[clm_list[len(clm_list)-1]].values
	return [x,y]
	
def mca_analysis(filename):
	cancer = pd.read_csv(filename)
	N_cols = cancer.shape[1]
	N_rows = cancer.shape[0]
	x_train = np.zeros((N_rows,1))
	y_train = np.zeros((N_rows,1))
	for i in np.arange(N_cols):
		current_col = np.transpose(np.array([cancer[cancer.columns[i]]]))
		vals, indices = np.unique(current_col, return_inverse=True)
		idx = np.transpose(np.array([indices]))
		n_cat = vals.max;
		if len(vals) > 1 and len(vals) < 11: # maximum of 10 categories per column
			if i < (N_cols-1):
				x_train = np.append(x_train, idx, 1)
			elif i == (N_cols-1):
				y_train = current_col
	print x_train
	x_train_2 = x_train[1:1000,1:30]
	#x_train_2 = np.random.rand(50,50)
	cancer_simple = pd.DataFrame(x_train_2,
								 index=x_train_2[:,0],
								 columns=x_train_2[0,0:])
	subcancer = cancer.ix[1:50,0:10]
	print "Subcancer"
	print x_train_2
	print np.isnan(subcancer).any()
	mca_counts = mca.mca(cancer_simple)
	print mca_counts.L

	
def score_random_forest(x_train, y_train, x_test, y_test,n_trees,minLeaf,maxDepth):
	clf = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=minLeaf,max_depth=maxDepth)
	clf.fit(x_train, y_train)
	return [clf.score(x_test,y_test), clf.score(x_train,y_train)]
	
def prob_random_forest(x_train, y_train, x_test, n_trees,minLeaf,maxDepth):
	clf = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=minLeaf,max_depth=maxDepth)
	clf.fit(x_train, y_train)
	return clf.predict_proba(x_test)
	
def score_adaboost(x_train, y_train, x_test, y_test,n_estimatorsAAA,learning_RRate,maxDepth):
	clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=n_estimatorsAAA, learning_rate=learning_RRate, algorithm="SAMME.R")
	clf.fit(x_train, y_train)
	return [clf.score(x_test,y_test), clf.score(x_train,y_train)]
	
def prob_adaboost(x_train, y_train, x_test, n_estimatorsAAA,learning_RRate,maxDepth):
	clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=n_estimatorsAAA, learning_rate=learning_RRate, algorithm="SAMME")
	clf.fit(x_train, y_train)
	return clf.predict_proba(x_test)

def random_forest_n_trees(x,y,folds):
	scoreRF = np.empty([folds,2])
	n_trees = [5,10,20,40,80]
	scoreRF_trees = np.empty([len(n_trees),2])
	for idx, trees in enumerate(n_trees):
		for i in range(folds):
			x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/folds, random_state=random.randint(1, 100))
			## Train Random Forest
			scoreRF[i,:] = score_random_forest(x_train, y_train,x_test, y_test, trees,1,1000)
		scoreRF_KFOLD = np.mean(scoreRF,axis=0)
		print "{} trees and {} error" .format( trees, scoreRF_KFOLD)
		scoreRF_trees[idx,:] = scoreRF_KFOLD
	plot_RF_n_trees(n_trees, scoreRF_trees)
	
	
def plot_RF_n_trees(x_tree, data):
	plt.plot(x_tree, data[:,0], label='validation')
	plt.plot(x_tree, data[:,1], label='train')
	plt.title('Test and train error for Random Forest. Miniproject 1')
	plt.xlabel('Number of trees')
	plt.ylabel('Accuracy in 10Fold Random Validation')
	plt.legend()
	plt.grid(True)
	plt.savefig("trees_n_hot.png")
	plt.show()
	
def random_forest_max_per_tree(x,y,folds):
	scoreRF = np.empty([folds,2])
	leafs = [1,2,4,8,16]
	scoreRF_trees = np.empty([len(leafs),2])
	for idx, minleaf in enumerate(leafs):
		for i in range(folds):
			x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/folds, random_state=random.randint(1, 100))
			## Train Random Forest
			scoreRF[i,:] = score_random_forest(x_train, y_train,x_test, y_test, 40,minleaf,1000)
		scoreRF_KFOLD = np.mean(scoreRF,axis=0)
		print "{} minleaf and {} error" .format( minleaf, scoreRF_KFOLD)
		scoreRF_trees[idx,:] = scoreRF_KFOLD
	plot_RF_minLeaf(leafs, scoreRF_trees)
	
def plot_RF_minLeaf(x_tree, data):
	plt.plot(x_tree, data[:,0], label='validation')
	plt.plot(x_tree, data[:,1], label='train')
	plt.title('Validation and train error for Random Forest. Miniproject 1')
	plt.xlabel('Min number of nodes per leaf')
	plt.ylabel('Accuracy in 10Fold Random Validation')
	plt.legend()
	plt.grid(True)
	plt.savefig("trees_max_per_hot.png")
	plt.show()
	
def random_forest_max_depth(x,y,folds):
	scoreRF = np.empty([folds,2])
	depth = [ 1,5,10,20,40,80,160,320]
	scoreRF_trees = np.empty([len(depth),2])
	for idx, depth_x in enumerate(depth):
		for i in range(folds):
			x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/folds, random_state=random.randint(1, 100))
			## Train Random Forest
			scoreRF[i,:] = score_random_forest(x_train, y_train,x_test, y_test, 40,8,depth_x)
		scoreRF_KFOLD = np.mean(scoreRF,axis=0)
		print "{} max depth and {} error" .format( depth_x, scoreRF_KFOLD)
		scoreRF_trees[idx,:] = scoreRF_KFOLD
	plot_RF_maxDepth(depth, scoreRF_trees)
	
def plot_RF_maxDepth(depth, data):
	plt.plot(depth, data[:,0], label='validation')
	plt.plot(depth, data[:,1], label='train')
	plt.title('Validation and train error for Random Forest. Miniproject 1')
	plt.xlabel('Max depth of the tree')
	plt.ylabel('Accuracy in 10Fold Random Validation')
	plt.legend()
	plt.grid(True)
	plt.savefig("trees_max_per_hot.png")
	plt.show()
	
	
def AB_n_est(x,y,folds):
	scoreRF = np.empty([folds,2])
	n_est = [ 10,20,40,80,160]
	scoreAB_trees = np.empty([len(n_est),2])
	for idx, n_estt in enumerate(n_est):
		for i in range(folds):
			x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/folds, random_state=random.randint(1, 100))
			## Train Random Forest
			scoreRF[i,:] = score_adaboost(x_train, y_train,x_test, y_test, n_estt,1,2)
		scoreRF_KFOLD = np.mean(scoreRF,axis=0)
		print "{} max depth and {} error" .format( n_estt, scoreRF_KFOLD)
		scoreAB_trees[idx,:] = scoreRF_KFOLD
	plot_AB_est(n_est, scoreAB_trees)
	
def plot_AB_est(n_est, data):
	plt.plot(n_est, data[:,0], label='validation')
	plt.plot(n_est, data[:,1], label='train')
	plt.title('Validation and train error for AdaBoost. Miniproject 1')
	plt.xlabel('Number of estimators')
	plt.ylabel('Accuracy in 10Fold Random Validation')
	plt.legend()
	plt.grid(True)
	plt.savefig("ab_est.png")
	plt.show()
	
def AB_learning_rate(x,y,folds):
	scoreRF = np.empty([folds,2])
	learning = [ 0.25,0.5,1,2,4,8]
	scoreAB_trees = np.empty([len(learning),2])
	for idx, x_learning in enumerate(learning):
		for i in range(folds):
			x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/folds, random_state=random.randint(1, 100))
			## Train Random Forest
			scoreRF[i,:] = score_adaboost(x_train, y_train,x_test, y_test, 40,x_learning,2)
		scoreRF_KFOLD = np.mean(scoreRF,axis=0)
		print "{} max depth and {} error" .format( x_learning, scoreRF_KFOLD)
		scoreAB_trees[idx,:] = scoreRF_KFOLD
	plot_AB_learning_rate(learning, scoreAB_trees)
	
def plot_AB_learning_rate(n_est, data):
	plt.plot(n_est, data[:,0], label='validation')
	plt.plot(n_est, data[:,1], label='train')
	plt.title('Validation and train error for AdaBoost. Miniproject 1')
	plt.xlabel('Learning Rate')
	plt.ylabel('Accuracy in 10Fold Random Validation')
	plt.legend()
	plt.grid(True)
	plt.savefig("ab_learning.png")
	plt.show()
	
	
def score_dnn (x_train, y_train, x_test, y_test,complexity):
	## Create your own model here given the constraints in the problem
	print x_train.shape
	model = Sequential()
	model.add(Dense(complexity,input_dim=x_train.shape[1]))
	model.add(Activation('relu'))
	model.add(Dropout(0.05))
	model.add(Dense(complexity/2))
	#model.add(Dropout(0.2))
	model.add(Activation('relu'))
	model.add(Dense(1))
	model.add(Activation('softmax'))
	model.summary()
	
	model.compile(loss='binary_crossentropy',optimizer='adadelta', metrics=['accuracy'])

	fit = model.fit(x_train, y_train, batch_size=32, nb_epoch=5, verbose=1)
	
	validation_error = score = model.evaluate(x_test, y_test, verbose=0)
	train_error = score = model.evaluate(x_train, y_train, verbose=0)
	return [validation_error[1], train_error[1]]
	
def dnn_complexity(x,y,folds):
	scoreRF = np.empty([folds,2])
	complexity_vector = [20,40,80,160]
	scoreDNN = np.empty([len(complexity_vector),2])
	for idx, complexity in enumerate(complexity_vector):
		for i in range(folds):
			x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/folds, random_state=random.randint(1, 100))
			## Train Random Forest
			scoreRF[i,:] = score_dnn(x_train, y_train,x_test, y_test, complexity)
			
		scoreRF_KFOLD = np.mean(scoreRF,axis=0)
		print scoreRF_KFOLD
		scoreDNN[idx,:] = scoreRF_KFOLD
	plot_dnn_complexity(complexity_vector, scoreDNN)
	
def plot_dnn_complexity(first_layer, data):
	plt.plot(first_layer, data[:,0], label='validation')
	plt.plot(first_layer, data[:,1], label='train')
	plt.title('Validation and train error for Neural Network. Miniproject 1')
	plt.xlabel('Size of the first layer')
	plt.ylabel('Accuracy in 10Fold Random Validation')
	plt.legend()
	plt.grid(True)
	plt.savefig("dnn_complexity.png")
	plt.show()
	
def save_histograms(x):
	idx = 0
	for xcolum in x.transpose():
		idx+=1
		plt.hist(xcolum)
		plt.savefig("hist_" + str(idx)+".png")
		plt.close()
		
def unique_columns(x):
	n_col = []
	for xcolum in x.transpose():
		n_col.append( len(np.unique(xcolum)) )
	return n_col
	
def hot_encode_input(x, n_col, col_max):
	x_hot = np.empty([x.shape[0],0])
	sizehot = 0
	for ncolum, xcolum in zip(n_col,x.T):
		if (ncolum<col_max and ncolum>1):
			sizehot = sizehot + ncolum
			unique_values = np.unique(xcolum)
			xcolum_hot = np.zeros([len(xcolum),ncolum])
			for xunit_id, xunit in enumerate(xcolum):
				for val_id, uniqueVal in enumerate(unique_values):
					if xunit == uniqueVal:
						xcolum_hot[xunit_id,val_id] = int(1)
					#else:
						#xcolum_hot[xunit_id,val_id] = int(0)
			#print xcolum_hot[0:20,:]
			#xcolum_hot = np_utils.to_categorical(xcolum,nb_classes=ncolum)
			x_hot = np.hstack((x_hot, xcolum_hot))
			#plt.hist(xcolum)
			#plt.show()
		elif (ncolum>1): 
			sizehot = sizehot + 1
			x_hot = np.hstack((x_hot, np.atleast_2d(xcolum).T))/max(xcolum)

	print "The data has been hot encode, the new size is {}" .format(sizehot)
	return x_hot
	
def RF_and_Adaboost_together(x,y,folds):
	score_RF = np.empty([folds,2])
	score_AB = np.empty([folds,2])
	score_together = np.empty([folds])
	for i in range(folds):
		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/folds, random_state=random.randint(1, 100))
		prob_RF = prob_random_forest(x_train, y_train,x_test, 100, 3, 100)
		score = score_random_forest(x_train, y_train,x_test,y_test, 100, 3, 100)
		score_RF[i,:] = score[0:2]
		clf = score[2]
		submit_solution(x,y,clf,str(score_RF[i,0]*1000)+"RF.csv")
		
		prob_AB = prob_adaboost(x_train, y_train,x_test, 50, 1, 100)
		score =  score_adaboost(x_train, y_train, x_test, y_test, 50, 1, 100)
		score_RF[i,:] = score[0:1]
		clf = score[2]
		submit_solution(x,y,clf,str(score_AB[i,0]*1000)+"AB.csv")
		
		prob_together = prob_RF + prob_AB # [np.max([prob_RF[:,0],prob_AB[:,0]],axis=1),np.max([prob_RF[:,1],prob_AB[:,1]],axis=1)] #multiply(prob_RF,prob_AB) 
		result_together = np.argmax(prob_together,axis=1)+1
		score_together[i] = accuracy_score(y_test,result_together)
		print "{} RF  and {} AB, {} together" .format( score_RF[i,:], score_AB[i,:], score_together[i])
		
	score_RF_KFOLD = np.mean(score_RF,axis=0)
	score_AB_KFOLD = np.mean(score_AB,axis=0)
	score_together_KFOLD = np.mean(score_together)
	print "Average {} RF  and {} AB, {} together" .format( score_RF_KFOLD, score_AB_KFOLD, score_together_KFOLD)
	return 0
	
#[ 0.76959951  0.80030928] RF  and [ 0.76697077  0.77415808] AB, 0.770372661203 together
#[ 0.77470233  0.80087629] RF  and [ 0.77810422  0.77341924] AB, 0.774702334931 together
#[ 0.76573373  0.8012543 ] RF  and [ 0.77114582  0.77512027] AB, 0.767125405907 together
#[ 0.77300139  0.80054983] RF  and [ 0.77114582  0.77278351] AB, 0.774393072522 together
#[ 0.78320705  0.79965636] RF  and [ 0.77810422  0.77261168] AB, 0.781042214319 together
#[ 0.76944487  0.79872852] RF  and [ 0.76836246  0.77233677] AB, 0.773156022885 together
#[ 0.77516623  0.79920962] RF  and [ 0.77733107  0.77345361] AB, 0.775939384568 together
#[ 0.76418741  0.80182131] RF  and [ 0.76635225  0.77376289] AB, 0.762950363383 together
#[ 0.7725375   0.79941581] RF  and [ 0.77624865  0.77316151] AB, 0.772537498067 together
#[ 0.77145508  0.79989691] RF  and [ 0.76882635  0.77378007] AB, 0.768980980362 together
#Average [ 0.77190351  0.80017182] RF  and [ 0.77225916  0.77345876] AB, 0.772119993815 together

	
def pca_analysis():
	xpca = PCA(n_components=50, svd_solver='randomized', whiten=True).fit_transform(x)
	PCA_A = PCA(n_components=50, svd_solver='randomized', whiten=True).fit(xpca)
	print np.diagonal(PCA_A.get_covariance())
	
def submit_solution(x,y,clf,name):
	x_test = read_test('test_2012.csv')
	clf.fit(x, y)
	y_test = clf.predict(x_test)
	write_solution(y_test,name)
	
def write_solution(y_test,name):
	with open(name, 'wb') as csvfile:
		spamwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
		spamwriter.writerow(['id'] + ['PES1'])
		for idx, value in enumerate(y_test):
			spamwriter.writerow([idx, value])
			
			
#def boost_RF(x,y):
	#init_columns = 10
	#N_cols = x.shape[1]
	#folds = 10
	#n_trees = 100
	#minLeaf = 5
	#maxDepth = 100
	#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/folds, random_state=init_state)
	#clf.
	#for n_columns in range(init_columns,N_cols):
		##
		#clf = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=minLeaf,max_depth=maxDepth)
		#clf.fit(x_train, y_train)
	    #score = clf.score(x_test,y_test)
	    #prob_old = 
		#prob = prob_random_forest(x_train, y_train, x_test, n_trees,minLeaf,maxDepth):
	
	
########################################################################
# ------- MINIPROJECT ---- MAIN CODE ----------------------------------#	
########################################################################

#mca_analysis('train_2008.csv')
[x, y] = read_data('train_2008.csv')
print "The data is size {} x {}" .format( x.shape[0], x.shape[1])


#col_max = 50
#n_col = unique_columns(x)
#x_hot = hot_encode_input(x, n_col, col_max)

# save_histograms(x)

# Split our data into training and validation
folds = 10

#random_forest_n_trees(x_hot,y,folds)
AB_learning_rate(x,y,folds)
#random_forest_max_depth(x_hot,y,folds)

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/folds, random_state=random.randint(1, 100))
#print "Random Forest {} accuracy ".format(score_random_forest(x_train, y_train, x_test, y_test,400,20,100))
#print "Adaboos {} accuracy ".format(score_adaboost(x_train, y_train, x_test, y_test,400,1,2))

#complexity = 100
#print score_dnn (x_train, y_train, x_test, y_test,complexity)

#dnn_complexity(x,y,folds)

#pca_analysis()

## Compare normal and hot input encoding
#init_state = random.randint(1, 100)
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/folds, random_state=init_state)
#scoreRF[i,:] = score_random_forest(x_train, y_train,x_test, y_test, 50,15,50)
	
#for i in range(folds):
	#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/folds, random_state=random.randint(1, 100))
	### Train Random Forest
	#scoreRF[i,:] = score_dnn(x_train, y_train,x_test, y_test, complexity)
#scoreRF_KFOLD = np.mean(scoreRF,axis=0)

#boost_RF(x,y)

#RF_and_Adaboost_together(x,y,folds)

#clf = RandomForestClassifier(n_estimators=1000, min_samples_leaf=5,max_depth=100)  #AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=300, learning_rate=1, algorithm="SAMME") #
name = "sub_11.csv"
#submit_solution(x,y,clf,name)
#submit_solution(x,y,clf,name)

#clfSVM = svm.LinearSVC(penalty='l1', loss='squared_hinge', dual=False,tol=1e-3)
#clfSVM.fit(x, y)
#print clfSVM.coef_

#clfSVM = svm.SVC(kernel='rbf')
#clfSVM.fit(x, y)
#print clfSVM.coef_

