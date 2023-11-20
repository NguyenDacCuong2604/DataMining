import weka.attributeSelection.InfoGainAttributeEval;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.util.*;


public class Main {
    public static final String pathSource = "datasource.arff";

    public static void main(String[] args) throws Exception {
        Instances data = DataSource.read(pathSource);
        System.out.println(data);

        //set Class Index
        int classIndex = data.numAttributes() - 1;
        data.setClassIndex(classIndex);

        //list name attribute
        String[] listName = getNameClasses(data);
        System.out.println(Arrays.toString(listName));

        //print values of class attribute
        Map<String, Integer> hashMap_class = probability_class(data);
        System.out.println(hashMap_class);
        //print values of attribute 1
        Map<String, Integer> hashMap_attribute1 = probabilities(data, 1);
        System.out.println(hashMap_attribute1);
        //entropy
        double entropy = entropy(data);
        System.out.println("Entropy: " + entropy);
        //info gain
        double infoGain = informationGain(data, 3, classIndex);
        System.out.println("InfoGain: " + infoGain);

        //so sanh
        InfoGainAttributeEval eval = new InfoGainAttributeEval();
        eval.buildEvaluator(data);
        double infoGainTest = eval.evaluateAttribute(3);
        System.out.println(infoGainTest);
    }

    public static String[] getNameClasses(Instances instances) {
        int numClasses = instances.numAttributes();
        String[] nameClasses = new String[numClasses];
        for (int i = 0; i < numClasses; i++) {
            nameClasses[i] = instances.attribute(i).name();
        }
        return nameClasses;
    }

    public static Map<String, Integer> probability_class(Instances instances) {
        Map<String, Integer> hashMap = new HashMap<>();
        Attribute attributeClass = instances.classAttribute();
        for (Instance instance : instances) {
            if (!hashMap.containsKey(instance.stringValue(attributeClass))) {
                hashMap.put(instance.stringValue(attributeClass), 1);
            } else
                hashMap.put(instance.stringValue(attributeClass), hashMap.get(instance.stringValue(attributeClass)) + 1);
        }
        return hashMap;
    }

    public static Map<String, Integer> probabilities(Instances instances, int attributeIndex) {
        Map<String, Integer> hashMap = new HashMap<>();
        Attribute classAttribute = instances.classAttribute();
        for (Instance instance : instances) {
            String name = format(instance.stringValue(attributeIndex), instance.stringValue(classAttribute));
            if (!hashMap.containsKey(name)) {
                hashMap.put(name, 1);
            } else {
                hashMap.put(name, hashMap.get(name) + 1);
            }
        }
        return hashMap;
    }

    public static String format(String attr, String classes) {
        return attr + "|" + classes;
    }

    public static double entropy(Instances instances) {
        Map<String, Integer> hashMap = probability_class(instances);
        double entropy = 0.0;
        for (String className : hashMap.keySet()) {
            double probability = (double) hashMap.get(className) / instances.numInstances();
            entropy -= (probability * (Math.log(probability) / Math.log(2)));
        }
        return entropy;
    }

    public static double informationGain(Instances instances, int attribute_Index, int class_index) {
        Attribute attribute = instances.attribute(attribute_Index);
        int numValueAttr = attribute.numValues();
        double entropyAttribute=0.0;
        for (int i = 0; i < numValueAttr; i++) {
            String value = attribute.value(i);
            int numberValue = getNumberValue(instances, attribute_Index, value);
            Map<String, Integer> map = getValuesAttribute(instances, attribute_Index, value);
            System.out.println(map);
            double entropy = 0.0;
            for(Integer valueOfKey : map.values()){
                double values = valueOfKey*1.0/numberValue;
                System.out.println(values);
                entropy -= values*(Math.log(values)/Math.log(2));
            }
            System.out.println(entropy);
            entropyAttribute+=(numberValue*1.0/instances.numInstances())*entropy;
        }
        return entropy(instances) - entropyAttribute;
    }

    public static int getNumberValue(Instances instances, int attributeIndex, String value) {
        int count = 0;
        for (Instance instance : instances) {
            if (instance.stringValue(instance.attribute(attributeIndex)).equals(value)) {
                count++;
            }
        }
        return count;
    }


    public static Map<String, Integer> getValuesAttribute(Instances instances, int attributeIndex, String valueAtr) {
        Map<String, Integer> map = new HashMap<>();
        for (Instance instance : instances) {
            String name = format(valueAtr, instance.stringValue(instance.classAttribute()));
            if (instance.stringValue(instance.attribute(attributeIndex)).equals(valueAtr)) {
                if (!map.containsKey(name)) {
                    map.put(name, 1);
                } else {
                    map.put(name, map.get(name) + 1);
                }
            }
        }
        return map;
    }
}