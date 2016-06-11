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

public class JSONLabelIterator implements LabelAwareIterator {
    private JSONArray lista;
    private Integer index = 0;
    protected LabelsSource labelsSource;

    /*
        Please keep this method protected, it's used in tests
     */
    protected JSONLabelIterator() {

    }

    public JSONLabelIterator(@NonNull JSONArray jsonArray) {
        try {

            fillLabels();
            this.lista = jsonArray;
        } catch (Exception var2) {
            throw new RuntimeException(var2);
        }
    }


    public JSONLabelIterator(@NonNull InputStream stream) {
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
        labels.add("0");
        labels.add("1");
        labels.add("2");
        labels.add("3");
        labels.add("4");
        labels.add("5");
        labels.add("6");
        labels.add("7");
        labels.add("8");
        labels.add("9");
        labels.add("10");
        labelsSource = new LabelsSource(labels);
    }

    public JSONLabelIterator(@NonNull String filePath) throws FileNotFoundException {
        this(new FileInputStream(filePath));
    }

    public boolean hasNextDocument() {
        return index < lista.length();
    }


    public LabelledDocument nextDocument() {
        Double nota = lista.getJSONObject(index).getDouble("nota");
        String texto = lista.getJSONObject(index).getString("texto");
        texto = texto.replaceAll("<BR>", " ");
        texto = texto.replaceAll("<br>", " ");
        index++;

        try {
            LabelledDocument document = new LabelledDocument();
            document.setContent(texto);
            int convNota = nota.intValue();
//            convNota = convNota - (convNota % 2);
            document.setLabel(String.valueOf(convNota));

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
