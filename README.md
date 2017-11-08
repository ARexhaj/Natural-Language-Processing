# Natural-Language-Processing
Using bag of words in Natural Language processing to see if reviews are positive or negative
I used a dataset of 1000 reviews to try and build a model to classify such reviews. 
I stripped the reviews of unnecessary words, built a bat of words to create a sparse matrix 
of all the relevant words in the reviews. Then I used a SVM model with a sigmoid kernel on 
the matrix to fit the model to the test
