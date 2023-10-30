package Bt1;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.bayes.NaiveBayes;

public class Naive_Bayes {
    public static void main(String[] args) throws Exception {
        //Load data
        Instances data = DataSource.read("D:\\Download\\bank.arff");
//        System.out.println(data);
        //index class
        data.setClassIndex(data.numAttributes()-1);

        Instances trainData = new Instances(data);
        Instances testData = new Instances(data);

//        // 7 - 3
//        int trainSize = (int) Math.round(data.numInstances() * 0.7);
//        int testSize = data.numInstances() - trainSize;
//
//        //split
//        trainData.randomize(new Random(1));
//        trainData = new Instances(trainData, 0, trainSize);
//        testData = new Instances(testData, trainSize, testSize);

        //create model
        NaiveBayes naiveBayes = new NaiveBayes();

        //train
        naiveBayes.buildClassifier(trainData);

        //evalution
        Evaluation evaluation = new Evaluation(trainData);
        evaluation.evaluateModel(naiveBayes, testData);

        System.out.println(evaluation.toSummaryString());


        System.out.println("Confusion matrix");
            //confusion matrix
        double[][] matrix = evaluation.confusionMatrix();
        for(int i = 0; i<matrix.length; i++){
            for(int j=0; j<matrix[i].length; j++){
                System.out.print(Math.round(matrix[i][j])+ " ");
            }
            System.out.println();
        }

//        int numInstances = trainData.numInstances();
//
//        for (int i = 0; i < numInstances; i++) {
//            Instance instance = testData.instance(i);
//            double actualClass = trainData.instance(i).classValue(); // Giá trị actual
//            double predictedClass = naiveBayes.classifyInstance(trainData.instance(i)); // predicted
//
//            System.out.print("Instance " + (i + 1) + " Actual = " + actualClass + ", Predicted = " + predictedClass+ " ");
//                double[] distribution = naiveBayes.distributionForInstance(instance); // Get the class probability distribution
//                System.out.print(" Prediction: ");
//                // Loop through the distribution to get the probabilities for each class
//                for (int j = 0; j < distribution.length; j++) {
//                    String className = testData.classAttribute().value(j);
//                    double classProbability = distribution[j];
//                    System.out.print(className + " = " + (classProbability * 100) + "%"+ "\t");
//                }
//                System.out.println();
//
//        }
    }
}
