import Genome.ANN;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;


/*
 * Classe contenant le main, va générer les espèces, les générations et les réseaux de neurones.
 */
public class Main {
    static int nbInputs = 2;
    static int nbOutputs = 1;
    static int nbMaxHiddenNodes = 100;

    static double threshold = 99; // Seuil en % à atteindre
    static int maxIterations = 20000; // Nombre d'itérations à faire au maximum
    static int populationSize = 200; // Taille fixe de la population, on aura toujours une population de cette taille

    static int nbMaxStable = 20; // Nombre d'itérations maximum où la population ne change pas
    static double C1 = 2; // Coefficients de la formule de distanciation
    static double C2 = 2;
    static double C3 = 1;
    static double thresholdDistanciation = 0.15; // Seuil de distanciation

    static int tauxMutation = 20; // Taux de mutation en %


    /*
     * Fonction principale, va générer les espèces, les générations et les réseaux de neurones.
     */
    public static void main(String[] args) {
        int[] entrees = new int[nbInputs];
        int[] sorties = new int[nbOutputs];

        // Au début, 1 espèce avec taille fixe de la population, totalement sans connexions
        Specie espece = GenerateFirstPopulation(populationSize);

        Neat neat = new Neat(espece, populationSize, nbMaxStable, C1, C2, C3, thresholdDistanciation, tauxMutation);

        System.out.println("\nDébut de l'apprentissage\n");

        ANN result = neat.learn(threshold, maxIterations);

        System.out.println("\nFin de l'apprentissage");

        double[] outputs = Neat.evaluateXor(result);

        DecimalFormatSymbols dfs = new DecimalFormatSymbols();
        dfs.setDecimalSeparator('.');
        DecimalFormat df = new DecimalFormat("0.00000000", dfs);

        System.out.println("\nRésultat : " + result);

        for (int i = 0; i < Neat.targetOutput.length; i++) {
            System.out.println("\tSortie " + i + " = " + df.format(outputs[i]) + "\t(" + Math.round(outputs[i]) + ") | Cible = " + Neat.targetOutput[i]);
        }
    }

    /*
     * Génère une population de départ, avec une espèce, et une population de taille
     * fixe, sans connexions
     */
    public static Specie GenerateFirstPopulation(int numberOfGenomes) {
        ArrayList<ANN> anns = new ArrayList<>();

        for (int x = 0; x < numberOfGenomes; x++) {
            ANN ann = new ANN(new ArrayList<>(), nbInputs, nbOutputs, nbMaxHiddenNodes);

            if (x == 0) { // Marquer le premier membre de l'espèce
                ann.first = true;
            }

            anns.add(ann);
        }

        return new Specie(C1, C2, C3, tauxMutation, anns);
    }
}
