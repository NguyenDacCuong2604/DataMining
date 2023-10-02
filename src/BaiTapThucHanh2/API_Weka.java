package BaiTapThucHanh2;

import java.util.List;

import weka.associations.Apriori;
import weka.associations.AssociationRule;
import weka.associations.FPGrowth;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class API_Weka {
	public static void main(String[] args) throws Exception {
		//Load data arff
		Instances data = DataSource.read("src/supermarket.arff");
//		System.out.println(data);
		
		//Apriori
		Apriori aprioriModel = new Apriori();
		//Minium Support
		aprioriModel.setLowerBoundMinSupport(0.5);
		//Minium confidence
		aprioriModel.setMinMetric(0.5);
		//Range Rules
		aprioriModel.setNumRules(5);
		long timeStart = System.currentTimeMillis();
		//Build
		aprioriModel.buildAssociations(data);
		long timeEnd = System.currentTimeMillis();
		long timeRun =timeEnd - timeStart;
		System.out.println("Time run Apriori: "+timeRun+"ms");
//		System.out.println(aprioriModel);
		List<AssociationRule> listRules = aprioriModel.getAssociationRules().getRules();
		printRules(listRules);
		
		System.out.println();
		
		//FP-Growth
		FPGrowth fpGrowthModel = new FPGrowth();
		//Minium Support
		fpGrowthModel.setLowerBoundMinSupport(0.5);
		//Minium confidence
		fpGrowthModel.setMinMetric(0.5);
		//Range Rules
		fpGrowthModel.setNumRulesToFind(5);
		long start = System.currentTimeMillis();
		//Build
		fpGrowthModel.buildAssociations(data);
		long end = System.currentTimeMillis();
		long run = end - start;
		System.out.println("Time run FP-Growth: "+run+"ms");
//		System.out.println(fpGrowthModel);
		List<AssociationRule> rules = fpGrowthModel.getAssociationRules().getRules();
		printRules(rules);
	}
	//Custom format Rules
	public static void printRules(List<AssociationRule> listRules) throws Exception {
		for(AssociationRule rule : listRules) {
			String formatRule = "";
			String premise = rule.getPremise().toString();
			String consequence = rule.getConsequence().toString();
			formatRule = premise + "--->" + consequence + "\n";
			String[] metricNames = rule.getMetricNamesForRule();
			for(String name : metricNames) {
				formatRule+="\t"+name+"="+rule.getNamedMetricValue(name)+"\n";
			}
			System.out.println(formatRule);
		}
	}
}
