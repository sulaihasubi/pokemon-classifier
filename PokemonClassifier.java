package global.skymind.solution.feedforward;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

/**Dataset is taken from Kaggle
 * https://www.kaggle.com/abcsds/pokemon
 */

public class PokemonClassifier {

    private static int epochs = 1000;

    public static void main(String[] args) throws IOException, InterruptedException {
        //create schema
        Schema inputDataSchema = new Schema.Builder()
                .addColumnInteger("#")
                .addColumnString("Name")
                .addColumnString("Type 1")
                .addColumnsString("Type 2")
                .addColumnInteger("Total")
                .addColumnsInteger("HP")
                .addColumnsInteger("Attack")
                .addColumnsInteger("Defense")
                .addColumnsInteger("Sp. Atk")
                .addColumnsInteger("Sp. Def")
                .addColumnsInteger("Speed")
                .addColumnsInteger("Generation")
                .addColumnCategorical("Legendary", Arrays.asList("True", "False"))
                .build();

        //create transform process
        Collection<String> columnsToRemove = new ArrayList<String>();
        columnsToRemove.add("#");
        columnsToRemove.add("Name");
        columnsToRemove.add("Type 2");

        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .stringToCategorical("Type 1",  Arrays.asList("Grass", "Fire", "Water", "Bug", "Normal", "Poison",
                        "Electric", "Ground", "Fairy", "Fighting", "Psychic", "Rock", "Ghost", "Ice", "Dragon", "Dark", "Steel", "Flying"))
                .categoricalToOneHot("Type 1")
                .categoricalToInteger("Legendary")
                .removeColumns(columnsToRemove)
                .build();

        //Now, print the schema after each time step (optional):
        int numActions = tp.getActionList().size();
        for(int i=0; i<numActions; i++ ){
            System.out.println("\n\n==================================================");
            System.out.println("-- Schema after step " + i + " (" + tp.getActionList().get(i) + ") --");

            System.out.println(tp.getSchemaAfterStep(i));
        }

        //After executing all of these operations, we have a new and different schema:
        Schema outputSchema = tp.getFinalSchema();

        System.out.println("\n\n\nSchema after transforming data:");
        System.out.println(outputSchema);

        //specify data location and create record reader
        File inputFile = new File(System.getProperty("user.home"), ".deeplearning4j/data/pokemon/Pokemon.csv");
        RecordReader rr = new CSVRecordReader(1, ',');
        rr.initialize(new FileSplit(inputFile));

        //Process the data:
        List<List<Writable>> originalData = new ArrayList<>();
        while(rr.hasNext()){
            originalData.add(rr.next());
        }

        List<List<Writable>> processedData = LocalTransformExecutor.execute(originalData, tp);

        //Create iterator from processedData
        RecordReader collectionRecordReader = new CollectionRecordReader(processedData);
        int labelIndex = outputSchema.getIndexOfColumn("Legendary");
        DataSetIterator iterator = new RecordReaderDataSetIterator(collectionRecordReader,1000,labelIndex,2);

        DataSet allData = iterator.next();

        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.8);
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        // normalize data to 0 - 1 and standardize it
        DataNormalization scaler = new NormalizerStandardize();
        scaler.fit(trainingData);
        scaler.transform(trainingData);
        scaler.transform(testData);

        int numInputs = (int) trainingData.getFeatures().shape()[1];
        System.out.println("numInputs: " + numInputs);
        int numOutputs = 2;

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .activation(Activation.LEAKYRELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.001))
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(50)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(50)
                        .nOut(50)
                        .build())
                .layer( new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(50)
                        .nOut(numOutputs)
                        .build())
                .build();


        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);
        model.setListeners(new ScoreIterationListener(50), new StatsListener(storage));

        for (int i=0; i<epochs; i++){
            model.fit(trainingData);
        }

        //evaluate the model on the test set
        Evaluation eval = new Evaluation(2);
        INDArray output = model.output(testData.getFeatures());
        eval.eval(testData.getLabels(), output);
        System.out.println(eval.stats());

    }
}
