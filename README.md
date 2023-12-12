# ML_classifiers

## KNN
- ` python knn.py -train [train file link] -test [test file link] -K [] -v `
- ` python knn.py -train "../knn_data/train1.csv" -test "../knn_data/test1.csv" -K 3 -v `
## Naive Bayes
- `python NaiveBayes.py -train [train file link] -test [test file link] -C [] -v`
- `python NaiveBayes.py -train "../NB_data/ex1_train.csv" -test "../NB_data/ex1_test.csv" -C 1 -v  `
## Kmeans
- `python NaiveBayes.py -train [train file link] -d [manh/e2] centroids`
- #### Manhattan Distance
`python kmeans.py -train "D:/D_Drive/college_classes/ai/lab/lab4/kmeans_data/input.txt" -d manh 0,0 200,200 500,500`
- #### Euclidean Distance
`python kmeans.py -train "D:/D_Drive/college_classes/ai/lab/lab4/kmeans_data/input.txt" -d e2 0,0 200,200 500,500`
