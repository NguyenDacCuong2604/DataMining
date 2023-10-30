package Bt2;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

import java.util.Random;

public class J48 {
    public static void main(String[] args) throws Exception {
        Instances dataset = ConverterUtils.DataSource.read("D:\\Download\\bank.arff");
        dataset.setClassIndex(dataset.numAttributes()-1);

        int numFolds = 2;
        Evaluation eval = new Evaluation(dataset);

        weka.classifiers.trees.J48 model = new weka.classifiers.trees.J48();
        eval.crossValidateModel(model, dataset, numFolds, new Random(1)); // Sử dụng seed 1 cho mục đích tái sản xuất

        // In các kết quả đánh giá
        System.out.println("Accuracy: " + eval.pctCorrect() + "%");
        System.out.println("Precision: " + eval.weightedPrecision());
        System.out.println("Recall: " + eval.weightedRecall());
        System.out.println("F1-Score: " + eval.weightedFMeasure());
    }
}
