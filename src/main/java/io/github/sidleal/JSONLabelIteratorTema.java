package io.github.sidleal;

import lombok.NonNull;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

public class JSONLabelIteratorTema implements LabelAwareIterator {
    private JSONArray lista;
    private Integer index = 0;
    protected LabelsSource labelsSource;

    /*
        Please keep this method protected, it's used in tests
     */
    protected JSONLabelIteratorTema() {

    }

    public JSONLabelIteratorTema(@NonNull JSONArray jsonArray) {
        try {

            fillLabels();
            this.lista = jsonArray;
        } catch (Exception var2) {
            throw new RuntimeException(var2);
        }
    }


    public JSONLabelIteratorTema(@NonNull InputStream stream) {
        try {

            fillLabels();
            JSONObject root = new JSONObject(IOUtils.toString(stream));
            this.lista = root.getJSONArray("redacoes");
        } catch (Exception var2) {
            throw new RuntimeException(var2);
        }
    }

    private void fillLabels() {
        List<String> labels = new ArrayList<String>();
        labels.add("Forma física, corpo perfeito e consumismo");
        labels.add("Impeachment: a presidente deve perder o mandato?");
        labels.add("Carta-convite: discutir discriminação na escola");
        labels.add("A tecnologia e a eliminação de empregos");
        labels.add("Por que o Brasil não consegue vencer o Aedes aegypti?");
        labels.add("Mariana: fatalidade ou negligência?");
        labels.add("Bandido bom é bandido morto?");
        labels.add("O sucesso vem da escola ou do esforço individual?");
        labels.add("Disciplina, ordem e autoridade favorecem a educação?");
        labelsSource = new LabelsSource(labels);
    }

    public JSONLabelIteratorTema(@NonNull String filePath) throws FileNotFoundException {
        this(new FileInputStream(filePath));
    }

    public boolean hasNextDocument() {
        return index < lista.length();
    }


    public LabelledDocument nextDocument() {
        Double nota = lista.getJSONObject(index).getDouble("nota");
        String texto = lista.getJSONObject(index).getString("texto");
        String tema = lista.getJSONObject(index).getString("tema");
        texto = texto.replaceAll("<BR>", " ");
        texto = texto.replaceAll("<br>", " ");
        index++;

        try {
            LabelledDocument document = new LabelledDocument();
            document.setContent(texto);
            int convNota = nota.intValue();
            if (convNota > 5) {
                convNota = 1;
            } else {
                convNota = 0;
            }
            document.setLabel(tema);

            return document;
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public void reset() {
        index = 0;
    }

    public LabelsSource getLabelsSource() {
        return labelsSource;
    }

}
