import Genome.ANN;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;


/*
 * Classe contenant le main, va générer les espèces, les générations et les réseaux de neurones.
 */
public class Main {
	static int nbInputs = 2;
	static int nbOutputs = 1;
	static int nbMaxHiddenNodes = 100000;

	static double threshold = 90; // Seuil en % à atteindre
	static int maxIterations = 20000; // Nombre d'itérations à faire au maximum
	static int taillePopulation = 200; // Taille fixe de la population, on aura toujours une population de cette taille

	static int nbMaxStable = 20; // Nombre d'itérations maximum où la population ne change pas
	static double C1 = 2; // Coefficients de la formule de distanciation
	static double C2 = 2;
	static double C3 = 0.4;
	static double thresholdDistanciation = 0.15; // Seuil de distanciation

	static int tauxMutation = 20; // Taux de mutation en %


/*
 * Fonction principale, va générer les espèces, les générations et les réseaux de neurones.
 */
	public static void main(String[] args) {


		System.out.println("Paramètres de l'algorithme :");

		int[] entrees = new int[nbInputs];
		System.out.print("\tEntrées : " + entrees.length + " ");
		System.out.println("(+ biais à 1)");

		int[] sorties = new int[nbOutputs];
		System.out.print("\tSorties : " + sorties.length + " ");

		System.out.println("\n\tNombre de noeuds cachés maximums : " + nbMaxHiddenNodes);

		System.out.println("\n\tTaille de la population : " + taillePopulation);
		System.out.println("\tNombre d'itérations maximum : " + maxIterations);
		System.out.println("\tSeuil à atteindre : " + threshold + " %");

		System.out.println("\n\tStabilité maximale d'une espèce : " + nbMaxStable);
		System.out.println("\tTaux de mutation : " + tauxMutation + " %");
		System.out.println("\tCoefficients de distanciation : " + C1 + ", " + C2 + ", " + C3);
		System.out.println("\tSeuil de distanciation : " + thresholdDistanciation);

		System.out.println("\n================\n");

		// Au début, 1 espèce avec taille fixe de la population, totalement sans connexions
		System.out.println("Génération d'une population de départ de " + taillePopulation + " ANNs vides dans une seule espèce");
		Specie espece = GenerateBeginPopulation(taillePopulation);

		Neat neat = new Neat(espece, taillePopulation, nbMaxStable, C1, C2, C3, thresholdDistanciation, tauxMutation);

		System.out.println("\nLancement de l'apprentissage ...");

		ANN result = neat.learn(threshold, maxIterations);

		System.out.println("\nFin de l'apprentissage");

		System.out.println("\n================\n");


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
	public static Specie GenerateBeginPopulation(int numberOfGenomes) {
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
