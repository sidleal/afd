package io.github.sidleal;

import lombok.NonNull;
import org.apache.commons.io.IOUtils;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.util.Iterator;


public class JSONIterator implements SentenceIterator, Iterable<String> {
    private SentencePreProcessor preProcessor;
    private JSONObject root;
    private JSONArray lista;
    private Integer index = 0;

    public JSONIterator(@NonNull File file) throws FileNotFoundException {
        this((InputStream) (new FileInputStream(file)));
    }

    public JSONIterator(@NonNull InputStream stream) {
        try {
            this.root = new JSONObject(IOUtils.toString(stream));
            this.lista = root.getJSONArray("redacoes");
        } catch (Exception var2) {
            throw new RuntimeException(var2);
        }
    }

    public JSONIterator(@NonNull String filePath) throws FileNotFoundException {
        this((InputStream) (new FileInputStream(filePath)));
    }

    public synchronized String nextSentence() {
        try {
            String texto = lista.getJSONObject(index).getString("texto");
            texto = texto.replaceAll("<BR>", " ");
            texto = texto.replaceAll("<br>", " ");
            texto = this.preProcessor != null ? this.preProcessor.preProcess(texto) : texto;
            index++;
            return texto;
        } catch (Exception var2) {
            throw new RuntimeException(var2);
        }
    }

    public synchronized boolean hasNext() {
        try {
            return index < this.lista.length();
        } catch (Exception var2) {
            return false;
        }
    }

    public synchronized void reset() {
        try {
            index = 0;
        } catch (Exception var2) {
            throw new RuntimeException(var2);
        }
    }

    public void finish() {
    }

    public SentencePreProcessor getPreProcessor() {
        return this.preProcessor;
    }

    public void setPreProcessor(SentencePreProcessor preProcessor) {
        this.preProcessor = preProcessor;
    }

    protected void finalize() throws Throwable {
        try {
        } catch (Exception var2) {
            var2.printStackTrace();
        }

        super.finalize();
    }

    public Iterator<String> iterator() {
        this.reset();
        Iterator ret = new Iterator() {
            public boolean hasNext() {
                return JSONIterator.this.hasNext();
            }

            public String next() {
                return JSONIterator.this.nextSentence();
            }

            public void remove() {
                throw new UnsupportedOperationException();
            }
        };
        return ret;
    }
}
