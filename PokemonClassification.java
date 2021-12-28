package global.skymind.solution.homework;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.arbiter.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;

public class PokemonClassification {
    private static int numLinesToSkip = 0;
    private static char delimiter = ',';

    private static int batchSize = 8000; // Pokomen data set: 800 examples total. We are loading all of them into one DataSet (not recommended for large data sets)
    private static int labelIndex = -1; // index of label/class column
    private static int numClasses = 2; // number of class in pokemon dataset where it is Legendary or not legendery

    private static int epochs = 2000;

    public static void main(String[] args) throws IOException, InterruptedException {
        // define csv file location
        File inputFile = new ClassPathResource("datavec/pokemonTransform.csv").getFile();
        FileSplit fileSplit = new FileSplit(inputFile);

        // get dataset using record reader. CSVRecordReader handles loading/parsing
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReader.initialize(fileSplit);

        // create iterator from record reader
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);
        DataSet allData = iterator.next();

        System.out.println("Shape of allData vector:");
        System.out.println(Arrays.toString(allData.getFeatures().shape()));

        // shuffle and split all data into training and test set
        allData.shuffle();
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.8);
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        // normalize data to 0 - 1 and standardize it
        DataNormalization scaler = new NormalizerStandardize();
        scaler.fit(trainingData);
        scaler.transform(trainingData);
        scaler.transform(testData);

        int numInputs = (int) trainingData.getFeatures().shape()[1];
        int numOutputs = 3;

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(0.1))
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(3)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(3)
                        .nOut(3)
                        .build())
                .layer( new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(3)
                        .nOut(numOutputs)
                        .build())
                .build();



        MultiLayerNetwork model = new MultiLayerNetwork(config);

        model.init();


        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        model.setListeners(new ScoreIterationListener(10));

        for (int i=0; i<epochs; i++){
            model.fit(trainingData);
        }

        //evaluate the model on the test set
        Evaluation eval = new Evaluation(3);
        INDArray output = model.output(testData.getFeatures());
        eval.eval(testData.getLabels(), output);
        System.out.println(eval.stats());

    }
}
