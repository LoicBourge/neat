package Genome;

/*
 * Classe faisant office de base de données des différentes innovations, afin de
 * ne pas devoir aller chercher dans les ConnectionGene.
 */
public class Innovation implements Comparable<Innovation> {
    public int innovation;
    public int input;
    public int output;

    /*
     * Constructeur de la classe Innovation, prend en paramètre l'entrée, la sortie
     * et le n° d'innovation.
     *
     */
    public Innovation(int innovation, int input, int output) {
        this.innovation = innovation;
        this.input = input;
        this.output = output;
    }

    @Override
    public int compareTo(Innovation compare) {
        return this.innovation - compare.innovation; // Tri ascendant
    }

    @Override
    public String toString() {
        return "(Innovation " + innovation + ", input : " + input + ", output : " + output + ")";
    }
}
