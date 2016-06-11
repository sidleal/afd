package io.github.sidleal;

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

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SentimentExampleIterator implements DataSetIterator {
    private final WordVectors wordVectors;
    private final int batchSize;
    private final int vectorSize;

    private int cursor = 0;
    private final TokenizerFactory tokenizerFactory;
    private JSONArray list;

    private static final Logger log = LoggerFactory.getLogger(SentimentExampleIterator.class);

    public SentimentExampleIterator(JSONArray list, WordVectors wordVectors, int batchSize) throws IOException {
        this.batchSize = batchSize;
        this.vectorSize = wordVectors.lookupTable().layerSize();

        this.list = list;
        this.wordVectors = wordVectors;

        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
    }


    public DataSet next(int num) {
        try{
            return nextDataSet(num);
        }catch(IOException e){
            throw new RuntimeException(e);
        }
    }

    private DataSet nextDataSet(int num) throws IOException {
        Double nota = list.getJSONObject(cursor).getDouble("nota");
        String texto = list.getJSONObject(cursor).getString("texto");
        texto = texto.replaceAll("<BR>", " ");
        texto = texto.replaceAll("<br>", " ");

        log.info("-------" + texto);

        cursor++;

        List<String> tokens = tokenizerFactory.create(texto).getTokens();
        List<String> tokensFiltered = new ArrayList<String>();
        for(String t : tokens ){
            if(wordVectors.hasWord(t)) {
                tokensFiltered.add(t);
            }
        }

        int length = tokensFiltered.size();

        INDArray features = Nd4j.create(1, vectorSize, length);

        INDArray labels = Nd4j.create(1, 5, length);
        INDArray featuresMask = Nd4j.zeros(1, length);
        INDArray labelsMask = Nd4j.zeros(1, length);

        int[] temp = new int[2];

        for( int j=0; j<tokensFiltered.size() && j<length; j++ ){
            String token = tokens.get(j);
            INDArray vector = wordVectors.getWordVectorMatrix(token);
            features.put(new INDArrayIndex[]{NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(j)}, vector);

            temp[0] = 0;
            temp[1] = j;
            featuresMask.putScalar(temp, 1.0);
        }

        int convNota = nota.intValue();
        convNota = convNota - (convNota % 2);
        if (convNota == 10) convNota = 8;

        labels.putScalar(new int[]{0,convNota,length-1},1.0);
        labelsMask.putScalar(new int[]{0,length-1},1.0);

        return new DataSet(features,labels,featuresMask,labelsMask);
    }

    public int totalExamples() {
        return list.length();
    }

    public int inputColumns() {
        return vectorSize;
    }

    public int totalOutcomes() {
        return 5;
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
        return Arrays.asList("0","2","4","6","8");
    }

    public boolean hasNext() {
        return cursor < numExamples();
    }

    public DataSet next() {
        return next(batchSize);
    }

    public void remove() {

    }

//    public String loadReviewToString(int index) throws IOException{
//        return FileUtils.readFileToString(f);
//    }

    /** Convenience method to get label for review */
//    public boolean isPositiveReview(int index){
//        return index%2 == 0;
//    }
}
