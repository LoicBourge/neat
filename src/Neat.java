import Genome.ANN;
import Genome.ConnectionGene;

import java.util.ArrayList;
import java.util.Collections;


/*
 * Classe principale de l'algorithme Neat, qui va lancer l'algorithme, retirer les
 * espèces stables et faire la distanciation.
 */
public class Neat {
	public static int[] targetOutput = new int[] { 0, 1, 1, 0 }; // La sortie du XOR désirée

	private final ArrayList<Specie> Species;
	private final int populationSize;
	private final int nbMaxStable; // Nombre d'espèces stables avant de retirer les moins performantes
	private final double C1; // Coefficients de la formule de distanciation
	private final double C2;
	private final double C3;
	private final double thresholdDistanciation; //
	private final int tauxMutation; // Taux de mutation en %


	/*
	 * Constructeur de la classe Neat
	 *	@param populationSize : Taille de la population
	 * @param nbMaxStable : Nombre d'espèces stables avant de retirer les moins performantes
	 * @param C1 : Coefficients de la formule de distanciation
	 * @param C2 : Coefficients de la formule de distanciation
	 * @param C3 : Coefficients de la formule de distanciation
	 * @param thresholdDistanciation
	 */
	public Neat(Specie specie, int populationSize, int nbMaxStable, double c1, double c2, double c3, double thresholdDistanciation, int tauxMutation) {
		Species = new ArrayList<>();
		Species.add(specie);
		this.populationSize = populationSize;
		this.nbMaxStable = nbMaxStable;
		C1 = c1;
		C2 = c2;
		C3 = c3;
		this.thresholdDistanciation = thresholdDistanciation;
		this.tauxMutation = tauxMutation;
	}

	// Apprentissage principal, ou se trouve la boucle principale
	public ANN learn(double threshold, int maxIterations) {
		double score = 0; // Fitness du meilleur génome
		int cpt = 0;
		ANN bestANN = null;


		// Boucler sur le score du meilleur génome tant que la précision < threshold (0.01) ou que l'on ait atteint le nombre d'itérations max
		while (score < threshold && cpt < maxIterations) {
			System.out.println("\nIteration " + cpt + " | Condition : " + score + " >= " + threshold);

			// Donc : Specie contient plusieurs Génomes, un Génome = ANN qui contient plusieurs Gènes, un Gene = NodeGene + ConnectionGene
			// Donc en gros, comme Specie est une espèce, elle possède plusieurs génomes (ANN), et donc à chaque fois qu'on doit calculer le fit et retirer les génomes faibles, on doit le faire sur des ANN

			for (Specie specie : Species) {
				specie.adjustedFit(); // Calcul du fit ajusté de chaque espèce
				specie.sortANNs(); // Trier les génomes du meilleur au pire
				specie.removeWeaks(); // Retirer génomes les plus faibles par espèce (garder que la première moitié)
			}

			// Retirer les espèces stables
			removeStables();

			int size = getPopulationSize();
			int numChildren = populationSize - size;
			ArrayList<Integer> childPerSpecies = getNumChildPerSpecies(numChildren);

			ArrayList<ANN> children = new ArrayList<>();

			// Boucler pour chaque espèce restante afin de faire des nouveaux children
			for (int i = 0; i < Species.size(); i++) {
				children.addAll(Species.get(i).crossANNs(childPerSpecies.get(i)));
			}

			// Mettre chaque enfant dans une espèce
			distanciation(children);

			// Calculer le fit de chaque espèce et trouver le génome ayant le score le plus élevé
			System.out.println("\tNombre d'espèces : " + Species.size());

			for (Specie specie : Species) {
				System.out.println("\t\t" + specie);
			}

			bestANN = calcBestFit();

			double[] outputs = evaluateXor(bestANN);
			bestANN.fit = fitXor(outputs);
			score = bestANN.fit;
			System.out.println("\tMeilleur ANN : " + bestANN);

			cpt++;
		}

		if (cpt == maxIterations) {
			System.out.println("\nNombre d'itérations maximum (" + maxIterations + ") atteint !");
		}
		else {
			System.out.println("\nScore suffisant (" + score + " sur " + threshold + ") atteint en " + cpt + " itérations !");
		}

		return bestANN;
	}

	// Méthode statique permettant d'évaluer un ANN selon le XOR
	public static double[] evaluateXor(ANN ann) {
		double[] outputs = new double[4];
		int cpt = 0;

		// Évaluer le réseau avec les 4 entrées possibles du XOR (00, 01, 10, 11)
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				float[] inputs = new float[] { i, j };
				outputs[cpt] = ann.evaluateNetwork(inputs)[0];
				cpt++;
			}
		}

		return outputs;
	}

	// Méthode statique permettant de calculer le fit selon le XOR à partir de sorties (obtenues avec evaluateXor)
	public static double fitXor(double[] outputs) {
		double fitness = 0;

		// Si 0 valeurs justes : fit = 0
		// 1 = 25
		// 2 = 50
		// 3 = 75
		// 4 = 100

		for (int i = 0; i < targetOutput.length; i++) {


			double calc = Math.abs(outputs[i] - (double) targetOutput[i]);

			if (calc <= 0.01) { // Ex : 0.5 - 0 = 0.5 <= 0.01 ?
				fitness += 25;
			}
			else if (calc <= 0.2) { // Ex : 0.5 - 0 = 0.5 <= 0.2 ?
				fitness += 12.5;
			}
			else if (calc <= 0.3) { // Ex : 0.5 - 0 = 0.5 <= 0.3 ?
				fitness += 8.33;
			}
			else if (calc <= 0.4) { // Ex : 0.5 - 0 = 0.5 <= 0.4 ?
				fitness += 6.25;
			}
			else if (calc <= 0.5) { // Ex : 0.5 - 0 = 0.5 <= 0.5 ?
				fitness += 5.0;
			}
		}

		return fitness;
	}

	// Va retirer les espèces stables
	private void removeStables() {
		ArrayList<Specie> aRetirer = new ArrayList<>();

		for (Specie specie : Species) {
			if (specie.getStableCounter() >= nbMaxStable && getBestFitSpecie() != specie) { // Garder un compteur pour chaque espèce, si l'espèce ne s'est pas améliorée après X générations et ne possède pas le meilleur fit, la retirer
				aRetirer.add(specie);
			}
			else if (specie.getANNs().size() == 0) { // Retirer les espèces vides
				aRetirer.add(specie);
			}
		}

		Species.removeAll(aRetirer);
	}

	// delta = (C1 * E) / N + (C2 * D) / N + C3 * W
	// C1, C2, C3 sont des coefficients à ajuster (dans exemple : 2, 2, 0.5)
	// N = nombre de gènes du génome le plus grand
	// E = nombre de gènes supplémentaires (excessifs)
	// D = nombre de gènes disjoints
	// W = différence de poids moyenne sur les gènes communs
	// Va répartir tous les enfants dans les bonnes espèces ou va en créer des nouvelles
	private void distanciation(ArrayList<ANN> enfants) {
		// Dans le site, il était mis de calculer à partir de ceux de l'ancienne génération, mais je fais avec la génération actuelle, afin que si plusieurs enfants sont similaires, ils aillent dans la même espèce au lieu de recréer plusieurs espèces

		ArrayList<Boolean> stables = new ArrayList<>();

		for (int i = 0; i < Species.size(); i++) {
			stables.add(false);
		}

		ArrayList<Double> deltas = new ArrayList<>();
		ArrayList<Integer> especes = new ArrayList<>();

		for (ANN enfant : enfants) {

			for (int i = 0; i < Species.size(); i++) {
				Specie espece = Species.get(i);

				// On teste la distance avec le premier membre de l'espèce (son créateur)
				ANN firstMembre = espece.getFirstANN();



				int n = espece.getLargerANN();

				int e = espece.excessElements(firstMembre.conGeneList, enfant.conGeneList).size();
				int d = espece.disjointsElements(firstMembre.conGeneList, enfant.conGeneList).size();

				ArrayList<ConnectionGene[]> genomesCommuns = espece.commonsElements(firstMembre.conGeneList, enfant.conGeneList);

				double w = 0;

				if (genomesCommuns.size() > 0) { // Pour éviter de faire une division par zéro pour la moyenne (d devenait NaN sinon)
					for (ConnectionGene[] genes : genomesCommuns) {
						w += (Math.abs(genes[0].getWeight() - genes[1].getWeight()));
					}

					w /= genomesCommuns.size();
				}

				double delta = 0;

				if (n > 0) { // Pour éviter de faire une division par zéro
					delta = (C1 * e) / n + (C2 * d) / n + C3 * w;
				}

				// IF delta > threshold --> Nouvelle espèce, ELSE on l'ajoute à la liste des espèces possibles
				if (delta <= thresholdDistanciation) {
					deltas.add(delta);
					especes.add(i);
				}
			}

			// Parcourir toutes les espèces et calculer leurs delta <= thresholdDistanciation, et mettre l'enfant dans celle qui a le meilleur. Si on n'a pas pû ajouter l'enfant, on crée une nouvelle espèce
			if (deltas.size() > 0) { // On a au moins une espèce dans laquelle on peut ajouter l'enfant
				double deltaMin = Collections.min(deltas); // Mettre l'enfant dans l'espèce qui a le delta minimum
				int index = especes.get(deltas.indexOf(deltaMin));
				Species.get(index).addANN(enfant);
				stables.set(index, true);
				Species.get(index).resetStableCounter(); // Reset le compteur de stabilité
			}
			else { // L'enfant n'a pû être ajouté à aucune des espèces, on en crée une nouvelle
				enfant.first = true; // Il crée l'espèce
				Specie nouvelle = new Specie(C1, C2, C3, tauxMutation, enfant);
				Species.add(nouvelle);
				stables.add(false);
			}
		}

		// Incrémenter le compteur de stabilité si l'espèce n'a pas du tout changée
		for (int i = 0; i < stables.size(); i++) {
			if (!stables.get(i)) {
				Species.get(i).incrementStableCounter();
			}
		}
	}

	// Calculer le fit de chaque espèce et trouver le génome ayant le score le plus élevé
	private ANN calcBestFit() {
		double bestFit = 0;
		ANN bestANN = null;

		for (Specie specie : Species) {
			specie.computeFit();

			ANN tmp = specie.bestANN();
			if (tmp.fit > bestFit) {
				bestFit = tmp.fit;
				bestANN = tmp;
			}
		}

		return bestANN;
	}

	// Obtenir l'espèce possédant le meilleur fit
	private Specie getBestFitSpecie() {
		Specie bestSpecie = Species.get(0);
		double bestFit = 0;

		for (Specie specie : Species) {
			specie.computeFit();

			double tmp = specie.bestFit();
			if (tmp > bestFit) {
				bestFit = tmp;
				bestSpecie = specie;
			}
		}

		return bestSpecie;
	}

	// Obtenir la taille totale de la population
	private int getPopulationSize() {
		int size = 0;

		for (Specie specie : Species) {
			size += specie.getANNs().size();
		}

		return size;
	}

	// Obtenir le nombre d'enfants que chaque espèce doit faire
	private ArrayList<Integer> getNumChildPerSpecies(int numChild) {
		ArrayList<Integer> childrenPerSpecies = new ArrayList<>();
		int total = 0;

		for (int i = 0; i < Species.size(); i++) {
			int nbre = numChild / Species.size();
			total += nbre;
			childrenPerSpecies.add(nbre);
		}

		// À cause des arrondis, il se peut qu'on n'ait pas exactement le nombre d'enfants à faire, donc on les rajoute
		if (total < numChild) {
			int nbre = numChild - total;

			for (int i = 0; i < childrenPerSpecies.size(); i++) {
				childrenPerSpecies.set(i, childrenPerSpecies.get(i) + 1);
				nbre--;

				if (nbre <= 0) {
					break;
				}
			}
		}

		return childrenPerSpecies;
	}
}
