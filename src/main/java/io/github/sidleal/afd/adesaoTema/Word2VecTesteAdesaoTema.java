package io.github.sidleal.afd.adesaoTema;

import org.apache.commons.collections.iterators.ListIteratorWrapper;
import org.apache.commons.collections.map.HashedMap;
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

public class Word2VecTesteAdesaoTema {

    private static final Logger log = LoggerFactory.getLogger(Word2VecTesteAdesaoTema.class);

    public static void main( String... args) throws Exception {
        for (int k = 0; k < 10; k++) {
            List<String[]> ret = new ArrayList<String[]>();
            for (int i = 0; i < 10; i++) {
                ret.addAll(run());
            }

            log.info("------------------------------------------------------- " + k);
            int acertos = 0;
            for (String[] item : ret) {
                log.info(item[0] + " --> " + item[1]);
                if (item[0].equals(item[1])) {
                    acertos++;
                }
            }

            log.info("total: " + ret.size());
            log.info("acertos: " + acertos);
            log.info("resultado: " + (acertos * 100.0 / ret.size()) + "%");
        }
    }

    public static List<String[]> run() throws Exception {

        JSONArray listaGeral = new JSONArray();
        for (int i = 1; i <= 9; i++) {
            String filePath = new ClassPathResource("corpus/uoleducacao_redacoes_" + String.format("%02d", i) + ".json").getFile().getAbsolutePath();
            JSONObject root = new JSONObject(IOUtils.toString(new FileInputStream(filePath)));
            JSONArray lista = root.getJSONArray("redacoes");
            String tema = root.getString("tema");

            for (int j = 0; j < lista.length(); j++) {
                JSONObject item = new JSONObject();
                Double nota = lista.getJSONObject(j).getDouble("nota");
                item.put("tema", tema);
                item.put("nota", nota);
                item.put("texto", lista.getJSONObject(j).getString("texto"));
                if (nota.intValue() > 3) {
                    listaGeral.put(item);
                }
            }
        }
        log.info("total redacoes: " + listaGeral.length());

        int listSize = listaGeral.length();
        List<Integer> samples = new ArrayList<Integer>();
        JSONArray listaTreino = new JSONArray();
        JSONArray listaTeste = new JSONArray();

        Map<String, Integer> summaryTreino = new HashMap<String, Integer>();
        Map<String, Integer> summaryTeste = new HashMap<String, Integer>();

        Map<String, List<Integer>> mapSum = new HashMap<String, List<Integer>>();
        for (int i = 0; i < listSize; i++) {
            JSONObject item = listaGeral.getJSONObject(i);
            String tema = item.getString("tema");
            if (mapSum.containsKey(tema)) {
                mapSum.get(tema).add(i);
            } else {
                List itens = new ArrayList<Integer>();
                itens.add(i);
                mapSum.put(tema, itens);
            }
        }

        for (List<Integer> mapItem : mapSum.values()) {
            int n = randomize(mapItem.size());
            samples.add(mapItem.get(n));
        }

        for (int i = 0; i < listSize; i++) {
            JSONObject item = listaGeral.getJSONObject(i);
            String tema = item.getString("tema");
            if (samples.contains(i)) {
                listaTeste.put(item);
                summarize(summaryTeste, tema);
            } else {
                listaTreino.put(item);
                summarize(summaryTreino, tema);
            }
        }

        log.info("Total de redacoes treino: " + listaTreino.length() + " - Teste: " + listaTeste.length());
        log.info("--------------------------------------------------------------------");
        log.info(padRight("Tema", 60) + " - " + padLeft("Treino", 15) + " - " + padLeft("Teste", 15));
        for (String key : summaryTreino.keySet()) {
            if (!summaryTeste.containsKey(key)) {
                summaryTeste.put(key, 0);
            }
            log.info(padRight(key, 60) + " - " + padLeft(summaryTreino.get(key).toString(), 15) + " - " + padLeft(summaryTeste.get(key).toString(), 15));
        }
        log.info("--------------------------------------------------------------------");

        LabelAwareIterator iterator = new JSONLabelIteratorTema(listaTreino);

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
        LabelAwareIterator unlabeledIterator = new JSONLabelIteratorTema(listaTeste);

        /*
         Now we'll iterate over unlabeled data, and check which label it could be assigned to
        */
        MeansBuilder meansBuilder = new MeansBuilder((InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable(), t);
        LabelSeeker seeker = new LabelSeeker(iterator.getLabelsSource().getLabels(), (InMemoryLookupTable<VocabWord>)  paragraphVectors.getLookupTable());

        List<String[]> ret = new ArrayList<String[]>();
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

                ret.add(new String[] {document.getLabel(), scores.get(0).getFirst()});
            }
        }
        return ret;
    }

    private static int randomize(Integer max) {
        Random rand = new Random();
        int  n = rand.nextInt(max);
    return n;
}

    private static void summarize(Map<String, Integer> map, String key) {
        if (map.containsKey(key)) {
            map.put(key, map.get(key) + 1);
        } else {
            map.put(key, 1);
        }
    }

    private static String padLeft(String s, int n) {
        return String.format("%1$" + n + "s", s);
    }

    private static String padRight(String s, int n) {
        return String.format("%1$-" + n + "s", s);
    }
}
