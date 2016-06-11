package io.github.sidleal;

/**
 * Created by lealsi on 10/06/2016.
 */
public class Testador {

    public static void main (String... args) {
        String ret = "blabla --> blabla\nblibli --> bloblo";
        String[] res = ret.split("\\n");
        for (String item : res) {
            String[] pares = item.split(" --> ");
            if (pares[0].equals(pares[1])) {
                System.out.println("aeeeeeee");
            }
        }
    }
}
