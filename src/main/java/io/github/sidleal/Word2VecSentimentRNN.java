package io.github.sidleal;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.io.IOUtils;
import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.json.JSONArray;
import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Word2VecSentimentRNN {

    private static final Logger log = LoggerFactory.getLogger(Word2VecSentimentRNN.class);

    public static void main(String[] args) throws Exception {

        int batchSize = 1;     //Number of examples in each minibatch
        int vectorSize = 500;   //Size of the word vectors.x
        int nEpochs = 1;        //Number of epochs (full passes of training data) to train on
        int truncateReviewsToLength = 500;  //Truncate reviews with length (# words) greater than this


        //Set up network configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .updater(Updater.RMSPROP)
                .regularization(true).l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
                .learningRate(0.0018)
                .list(2)
                .layer(0, new GravesLSTM.Builder().nIn(vectorSize).nOut(200)
                        .activation("softsign").build())
                .layer(1, new RnnOutputLayer.Builder().activation("softmax")
                        .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(200).nOut(2).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        String filePathVec = new ClassPathResource("mariana_vec2.txt").getFile().getAbsolutePath();
        WordVectors wordVectors = WordVectorSerializer.loadTxtVectors(new File(filePathVec));


        JSONArray listaGeral = new JSONArray();
        String filePath = new ClassPathResource("corpus/uoleducacao_redacoes_07.json").getFile().getAbsolutePath();
        JSONObject root = new JSONObject(IOUtils.toString(new FileInputStream(filePath)));
        JSONArray lista = root.getJSONArray("redacoes");

        for (int j = 0; j < lista.length(); j++) {
            JSONObject item = new JSONObject();
            item.put("nota", lista.getJSONObject(j).getDouble("nota"));
            item.put("texto", lista.getJSONObject(j).getString("texto"));
            listaGeral.put(item);
        }

        Random rand = new Random();
        int listSize = listaGeral.length();
        int sampleSize = listSize * 30 / 100;
        List<Integer> samples = new ArrayList<Integer>();
        for (int i = 0; i < sampleSize; i++) {
            int  n = rand.nextInt(listSize);
            samples.add(n);
        }

        JSONArray listaTreino = new JSONArray();
        JSONArray listaTeste = new JSONArray();

        for (int i = 0; i < listSize; i++) {
            JSONObject item = listaGeral.getJSONObject(i);
            if (samples.contains(i)) {
                listaTeste.put(item);
            } else {
                listaTreino.put(item);
            }
        }

        log.info("Total de redacoes treino: " + listaTreino.length() + " - Teste: " + sampleSize);

        DataSetIterator train = new AsyncDataSetIterator(new SentimentExampleIterator2(listaTreino,wordVectors,batchSize,truncateReviewsToLength),1);
        DataSetIterator test = new AsyncDataSetIterator(new SentimentExampleIterator2(listaTeste,wordVectors,batchSize,truncateReviewsToLength),1);


        System.out.println("Starting training");
        for( int i=0; i<nEpochs; i++ ){
            net.fit(train);
            train.reset();
            System.out.println("Epoch " + i + " complete. Starting evaluation:");

            Evaluation evaluation = new Evaluation();
            while(test.hasNext()){
                DataSet t = test.next();
                INDArray features = t.getFeatureMatrix();
                INDArray labels = t.getLabels();
                INDArray inMask = t.getFeaturesMaskArray();
                INDArray outMask = t.getLabelsMaskArray();
                INDArray predicted = net.output(features,false,inMask,outMask);

                evaluation.evalTimeSeries(labels,predicted,outMask);
            }
            test.reset();

            System.out.println(evaluation.stats());
        }


        System.out.println("----- Example complete -----");
    }

}
