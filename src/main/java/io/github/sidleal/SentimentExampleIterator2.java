package io.github.sidleal;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.json.JSONArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;

public class SentimentExampleIterator2 implements DataSetIterator {

    private static final Logger log = LoggerFactory.getLogger(SentimentExampleIterator2.class);

    private final WordVectors wordVectors;
    private final int batchSize;
    private final int vectorSize;
    private final int truncateLength;
    private JSONArray list;

    private int cursor = 0;
    private final TokenizerFactory tokenizerFactory;

    public SentimentExampleIterator2(JSONArray list, WordVectors wordVectors, int batchSize, int truncateLength) throws IOException {
        this.batchSize = batchSize;
        this.vectorSize = wordVectors.lookupTable().layerSize();
        this.list = list;

        this.wordVectors = wordVectors;
        this.truncateLength = truncateLength;

        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
    }


    public DataSet next(int num) {
        if (cursor >= totalExamples()) throw new NoSuchElementException();
        try{
            return nextDataSet(num);
        }catch(IOException e){
            throw new RuntimeException(e);
        }
    }

    private DataSet nextDataSet(int num) throws IOException {
        List<String> reviews = new ArrayList<String>(num);
        boolean[] positive = new boolean[num];
        for( int i=0; i<num && cursor<totalExamples(); i++ ){

            Double nota = list.getJSONObject(cursor).getDouble("nota");
            String texto = list.getJSONObject(cursor).getString("texto");
            texto = texto.replaceAll("<BR>", " ");
            texto = texto.replaceAll("<br>", " ");

            reviews.add(texto);
            int convNota = nota.intValue();
            if (convNota > 5) {
                positive[i] = true;
            } else {
                positive[i] = false;
            }

            //log.info("-------" + convNota + " -- " + positive[i] + " -- " + texto);

            cursor++;
        }

        //Second: tokenize reviews and filter out unknown words
        List<List<String>> allTokens = new ArrayList<List<String>>(reviews.size());
        int maxLength = 0;
        for(String s : reviews){
            List<String> tokens = tokenizerFactory.create(s).getTokens();
            List<String> tokensFiltered = new ArrayList<String>();
            for(String t : tokens ){
                if(wordVectors.hasWord(t)) {
                    tokensFiltered.add(t);
                } else {
                    tokensFiltered.add("-");
                }
            }
            allTokens.add(tokensFiltered);
            String bla = "";
            for (String a : tokensFiltered) {
                bla += a + " ";
            }
            log.info("----------" + bla);
            maxLength = Math.max(maxLength,tokensFiltered.size());
        }

        //If longest review exceeds 'truncateLength': only take the first 'truncateLength' words
        if(maxLength > truncateLength) maxLength = truncateLength;

        //Create data for training
        //Here: we have reviews.size() examples of varying lengths
        INDArray features = Nd4j.create(reviews.size(), vectorSize, maxLength);
        INDArray labels = Nd4j.create(reviews.size(), 2, maxLength);    //Two labels: positive or negative
        //Because we are dealing with reviews of different lengths and only one output at the final time step: use padding arrays
        //Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
        INDArray featuresMask = Nd4j.zeros(reviews.size(), maxLength);
        INDArray labelsMask = Nd4j.zeros(reviews.size(), maxLength);

        int[] temp = new int[2];
        for( int i=0; i<reviews.size(); i++ ){
            List<String> tokens = allTokens.get(i);
            temp[0] = i;
            //Get word vectors for each word in review, and put them in the training data
            for( int j=0; j<tokens.size() && j<maxLength; j++ ){
                String token = tokens.get(j);
                INDArray vector = wordVectors.getWordVectorMatrix(token);
                features.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j)}, vector);

                temp[1] = j;
                featuresMask.putScalar(temp, 1.0);  //Word is present (not padding) for this example + time step -> 1.0 in features mask
            }

            int idx = (positive[i] ? 1 : 0);
            int lastIdx = Math.min(tokens.size(),maxLength);
            labels.putScalar(new int[]{i,idx,lastIdx-1},1.0);   //Set label: [0,1] for negative, [1,0] for positive
            labelsMask.putScalar(new int[]{i,lastIdx-1},1.0);   //Specify that an output exists at the final time step for this example
        }

        return new DataSet(features,labels,featuresMask,labelsMask);
    }

    public int totalExamples() {
        return list.length();
    }

    public int inputColumns() {
        return vectorSize;
    }

    public int totalOutcomes() {
        return 2;
    }

    public void reset() {
        cursor = 0;
    }

    public int batch() {
        return batchSize;
    }

    public int cursor() {
        return cursor;
    }

    public int numExamples() {
        return totalExamples();
    }

    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException();
    }

    public List<String> getLabels() {
        return Arrays.asList("positive","negative");
    }

    public boolean hasNext() {
        return cursor < numExamples();
    }

    public DataSet next() {
        return next(batchSize);
    }

    public void remove() {

    }

//    /** Convenience method for loading review to String */
//    public String loadReviewToString(int index) throws IOException{
//        return list.getJSONObject(index).getString("texto");
//    }
//
//    /** Convenience method to get label for review */
//    public boolean isPositiveReview(int index){
//        return false;
//    }
}
