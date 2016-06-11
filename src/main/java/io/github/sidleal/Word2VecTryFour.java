package io.github.sidleal;

import org.apache.commons.io.IOUtils;
import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.json.JSONArray;
import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileInputStream;
import java.util.*;

public class Word2VecTryFour {

    private static final Logger log = LoggerFactory.getLogger(Word2VecTryFour.class);

    public static void main( String... args) throws Exception {
        String ret = "";
        for (int i = 0; i < 10; i++) {
            ret += run();
        }

        log.info("-------------------------------------------------------");
        log.info(ret);
    }

    public static String run() throws Exception {

        JSONArray listaGeral = new JSONArray();
        for (int i = 1; i <= 9; i++) {
            String filePath = new ClassPathResource("corpus/uoleducacao_redacoes_0" + i + ".json").getFile().getAbsolutePath();
            JSONObject root = new JSONObject(IOUtils.toString(new FileInputStream(filePath)));
            JSONArray lista = root.getJSONArray("redacoes");

            for (int j = 0; j < lista.length(); j++) {
                JSONObject item = new JSONObject();
                item.put("nota", lista.getJSONObject(j).getDouble("nota"));
                item.put("texto", lista.getJSONObject(j).getString("texto"));
                listaGeral.put(item);
            }
        }

        Random rand = new Random();
        int listSize = listaGeral.length();
        int sampleSize = listSize * 10 / 100;
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

        LabelAwareIterator iterator = new JSONLabelIteratorMq(listaTreino);

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        // ParagraphVectors training configuration
        ParagraphVectors paragraphVectors = new ParagraphVectors.Builder()
                .learningRate(0.025)
                .minLearningRate(0.001)
                .batchSize(1000)
                .epochs(20)
                .iterate(iterator)
                .trainWordVectors(true)
                .tokenizerFactory(t)
                .build();

        // Start model training
        paragraphVectors.fit();


        /*
         At this point we assume that we have model built and we can check, which categories our unlabeled document falls into
         So we'll start loading our unlabeled documents and checking them
        */
        LabelAwareIterator unlabeledIterator = new JSONLabelIteratorMq(listaTeste);

        /*
         Now we'll iterate over unlabeled data, and check which label it could be assigned to
        */
        MeansBuilder meansBuilder = new MeansBuilder((InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable(), t);
        LabelSeeker seeker = new LabelSeeker(iterator.getLabelsSource().getLabels(), (InMemoryLookupTable<VocabWord>)  paragraphVectors.getLookupTable());

        String ret = "";
        while (unlabeledIterator.hasNextDocument()) {
            LabelledDocument document = unlabeledIterator.nextDocument();

            INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
            if (documentAsCentroid != null) {
                List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);

                Collections.sort(scores, new Comparator<Pair<String, Double>>() {
                    public int compare(Pair<String, Double> o1, Pair<String, Double> o2) {
                        return o2.getSecond().compareTo(o1.getSecond());
                    }
                });

                log.info("Document '" + document.getLabel() + "' falls into the following categories: ");
                log.info("        " + scores.get(0).getFirst() + ": " + scores.get(0).getSecond());
                log.info("        " + scores.get(1).getFirst() + ": " + scores.get(1).getSecond());

                ret += document.getLabel() + " --> " + scores.get(0).getFirst() + "\n";
            }
        }
        return ret;
    }
}