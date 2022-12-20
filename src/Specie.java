import Genome.ANN;
import Genome.ConnectionGene;
import Genome.Innovation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

// Classe contenant une espèce, qui va contenir plusieurs individus (des ANN). Va faire les calculs des fits, retirer les génomes faibles et toute la reproduction.

public class Specie {
	private final double C1; // Coefficients de la formule de distanciation
	private final double C2;
	private final double C3;
	private final int tauxMutation; // Taux de mutation en %

	private static final AtomicInteger SpecieCounter = new AtomicInteger(); // Pour que chaque espèce ait un numéro unique

	private int stableCounter;  // Compteur de générations où l'espèce n'a pas changé
	private final int specieNumber;  // Numéro de l'espèce
	private int globalInnovation;  // Numéro d'innovation global
	private ArrayList<ANN> ANNs;  // Liste des ANN de l'espèce
	private double[] fits; // Fits de chaque génome
	private double[] adjustedFitness; // Fits ajustés de chaque génome


	/*
	 * Constructeur de l'espèce
	 * @param C1 Coefficient de la formule de distanciation
	 * 		   C2 Coefficient de la formule de distanciation
	 * 		   C3 Coefficient de la formule de distanciation
	 * 		   tauxMutation Taux de mutation en %
	 */
	private Specie(double c1, double c2, double c3, int tauxMutation) {
		stableCounter = 0;
		specieNumber = SpecieCounter.incrementAndGet();
		adjustedFitness = new double[0];
		globalInnovation = 0;
		C1 = c1;
		C2 = c2;
		C3 = c3;
		this.tauxMutation = tauxMutation;
	}


	/*
	 * Constructeur de l'espèce
	 * @param C1 Coefficient de la formule de distanciation
	 * 		   C2 Coefficient de la formule de distanciation
	 * 		   C3 Coefficient de la formule de distanciation
	 * 		   tauxMutation Taux de mutation en %
	 * 		   ANNs Liste des ANN de l'espèce
	 */
	public Specie(double c1, double c2, double c3, int tauxMutation, ArrayList<ANN> anns) {
		this(c1, c2, c3, tauxMutation);
		ANNs = anns;
	}

	/*
	 * Constructeur de l'espèce
	 * @param C1 Coefficient de la formule de distanciation
	 * 		   C2 Coefficient de la formule de distanciation
	 * 		   C3 Coefficient de la formule de distanciation
	 * 		   tauxMutation Taux de mutation en %
	 * 		   ANN ANN de l'espèce
	 */
	public Specie(double c1, double c2, double c3, int tauxMutation, ANN ann) {
		this(c1, c2, c3, tauxMutation);
		ANNs = new ArrayList<>();
		ANNs.add(ann);
	}

	public int getStableCounter() {
		return stableCounter;
	}

	public void incrementStableCounter() {
		stableCounter++;
	}

	public void resetStableCounter() {
		stableCounter = 0;
	}

	public int getSpecieNumber() {
		return specieNumber;
	}

	public ArrayList<ANN> getANNs() {
		return ANNs;
	}

	public void setANNs(ArrayList<ANN> ANNs) {
		this.ANNs = ANNs;
	}

	public void addANN(ANN ann) {
		this.ANNs.add(ann);
	}

	public double[] getAdjustedFitness() {
		return adjustedFitness;
	}

	public int getGlobalInnovation() {
		return globalInnovation;
	}
	/*
	 * Calcul des fits de chaque ANN de l'espèce
	 * @param fitnessFunction Fonction de fitness
	 */
	public double[] computeFit() {
		fits = new double[ANNs.size()];

		for (int i = 0; i < ANNs.size(); i++) {
			double[] outputs = Neat.evaluateXor(ANNs.get(i));
			fits[i] = Neat.fitXor(outputs);
			ANNs.get(i).fit = fits[i];
		}

		return fits;
	}

	// Va retourner le meilleur fit de tous les génomes
	public double bestFit() {
		double max = 0;

		for (double fit : fits) {
			if (fit > max) {
				max = fit;
			}
		}

		return max;
	}

	// Va retourner l'ANN possédant le meilleur fit
	public ANN bestANN() {
		double max = 0;
		ANN bestANN = null;

		for (ANN ann : ANNs) {
			if (ann.fit > max) {
				max = ann.fit;
				bestANN = ann;
			}
		}

		return bestANN;
	}

	// Va calculer le fit ajusté de l'espèce
	public void adjustedFit() {
		// f' = f / #genomes (avec #genomes le nombre d'ANN) le tout pour chaque espèce
		adjustedFitness = new double[ANNs.size()];

		// Calculer le fit de chaque génome
		adjustedFitness = computeFit();

		// Calculer le fit ajusté
		for (int i = 0; i < ANNs.size(); i++) {
			adjustedFitness[i] /= ANNs.size();
		}

		//System.out.println("Fits ajustés : " + Arrays.toString(adjustedFitness));
	}

	// Trier les génomes (ANNs) selon leurs fits (du meilleur au pire)
	public void sortANNs() {
		ANNs.sort(ANN::compareTo);
	}

	// Va retirer les génomes les plus faibles (ne garde que la première moitié)
	public void removeWeaks() {
		if (ANNs.size() > 1) { // Faire en sorte que l'espèce ait toujours au moins un seul enfant
			int half = ANNs.size() / 2 + (ANNs.size() % 2); // Obtenir la moitié

			ANN oldFirst = getFirstANN();

			ArrayList<ANN> aRetirer = new ArrayList<>();

			for (int i = 0; i < ANNs.size(); i++) {
				if (i >= half) {
					aRetirer.add(ANNs.get(i));
				}
			}

			ANNs.removeAll(aRetirer);

			// Si le first n'est plus dans la liste, on recalcule un nouveau first
			if (!ANNs.contains(oldFirst)) {
				calcFirstANN(oldFirst);
			}
		}
	}

	// Permet de calculer l'ANN qui deviendra le nouveau membre représentatif de l'espèce
	public void calcFirstANN(ANN firstMembre) {
		// On teste la distance avec le premier membre de l'espèce (son créateur)
		ArrayList<Double> deltas = new ArrayList<>();
		ArrayList<Integer> genomes = new ArrayList<>();

		for (int i = 0; i < ANNs.size(); i++) {
			ANN genomeAComparer = ANNs.get(i);

			// delta = (C1 * E) / N + (C2 * D) / N + C3 * W
			// C1, C2, C3 sont des coefficients à ajuster (dans exemple : 2, 2, 0.5)
			// N = nombre de gènes du génome le plus grand
			// E = nombre de gènes supplémentaires (excessifs)
			// D = nombre de gènes disjoints
			// W = différence de poids moyenne sur les gènes communs

			int n = getLargerANN();

			int e = excessElements(firstMembre.conGeneList, genomeAComparer.conGeneList).size();
			int d = disjointsElements(firstMembre.conGeneList, genomeAComparer.conGeneList).size();

			ArrayList<ConnectionGene[]> genomesCommuns = commonsElements(firstMembre.conGeneList, genomeAComparer.conGeneList);

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

			//System.out.println("Distanciation : n = " + n + " | e = " + e + " | d = " + d + " | w = " + w);
			//System.out.println("Delta = " + delta);
			//System.out.println("Threshold = " + thresholdDistanciation);

			deltas.add(delta);
			genomes.add(i);
		}

		// Un génome est meilleur que le first, il devient le first
		double deltaMin = Collections.min(deltas); // Mettre l'enfant dans l'espèce qui a le delta minimum
		int index = genomes.get(deltas.indexOf(deltaMin));
		ANNs.get(index).first = true;
		//System.out.println("ANN " + ANNs.get(index) + " devient le first !");
		// Pas la peine d'enlever le first, car il sera remove
	}

	// Retourne les génomes communs entre les 2 listes (ceux ayant le même n° d'innovation)
	public ArrayList<ConnectionGene[]> commonsElements(ArrayList<ConnectionGene> genomes1, ArrayList<ConnectionGene> genomes2) {
		ArrayList<ConnectionGene[]> rslt = new ArrayList<>();

		// Comparer les 2 listes afin de voir ce qui est en commun
		for (ConnectionGene genome1 : genomes1) {
			for (ConnectionGene genome2 : genomes2) {
				if (genome1.getInnovation() == genome2.getInnovation()) { // Si les 2 parents ont des gènes ayant le même n° d'innovation
					rslt.add(new ConnectionGene[] { genome1, genome2 });
				}
			}
		}

		return rslt;
	}

	// Va retourner le plus grand n° d'innovation de tous les génomes ConnectionGene
	private double bestConnectionInnovation(ArrayList<ConnectionGene> genomes) {
		double max = genomes.get(0).getInnovation();

		for (int i = 1; i < genomes.size(); i++) {
			if (genomes.get(i).getInnovation() > max) {
				max = genomes.get(i).getInnovation();
			}
		}

		return max;
	}

	// Va retirer les génomes communs aux 2, et retourner une liste contenant que ceux qui ne sont pas communs
	private ArrayList<ConnectionGene> removeCommonsElements(ArrayList<ConnectionGene> genomes1, ArrayList<ConnectionGene> genomes2) {
		// Voir les éléments ayant des n° d'innovations communs et les retirer de la 2e liste
		ArrayList<ConnectionGene[]> communs = commonsElements(genomes1, genomes2);
		ArrayList<ConnectionGene> com = new ArrayList<>();
		ArrayList<ConnectionGene> rslt = new ArrayList<>(genomes2); // Nouvelle liste sinon on effaçait la liste des génomes du parent !

		for (ConnectionGene[] genomes : communs) {
			com.add(genomes[1]);
		}

		rslt.removeAll(com); // La liste contient maintenant que les éléments qui ne sont pas dans la première liste fournie
		return rslt;
	}

	// Retourne les génomes présents dans la 2e liste qui on un n° d'innovation compris dans le range de ceux de la première liste
	public ArrayList<ConnectionGene> disjointsElements(ArrayList<ConnectionGene> genomes1, ArrayList<ConnectionGene> genomes2) {
		ArrayList<ConnectionGene> rslt = new ArrayList<>(); // Nouvelle liste sinon on effaçait la liste des génomes du parent !

		// On récupère le meilleur n° d'innovation
		if (genomes1 != null && genomes1.size() > 0) {
			double maxInnovation = bestConnectionInnovation(genomes1);

			rslt = removeCommonsElements(genomes1, genomes2);

			// On retire tous les éléments qui ont un n° d'innovation < que le n° d'innovation maximum de la première liste (donc on retire ceux qui sont disjoints)
			rslt.removeIf(innovation -> innovation.getInnovation() < maxInnovation); // RemoveIf car sinon en bouclant pour retirer lançait une java.util.ConcurrentModificationException
		}

		return rslt;
	}

	// Retourne les génomes présents dans la 2e liste qui on un n° d'innovation plus grand que ceux de la première liste
	public ArrayList<ConnectionGene> excessElements(ArrayList<ConnectionGene> genomes1, ArrayList<ConnectionGene> genomes2) {
		ArrayList<ConnectionGene> rslt = new ArrayList<>(); // Nouvelle liste sinon on effaçait la liste des génomes du parent !

		// On récupère le meilleur n° d'innovation
		if (genomes1 != null && genomes1.size() > 0) {
			double maxInnovation = bestConnectionInnovation(genomes1);

			rslt = removeCommonsElements(genomes1, genomes2);

			// On retire tous les éléments qui ont un n° d'innovation > que le n° d'innovation maximum de la première liste (donc on retire ceux qui sont excessifs)
			rslt.removeIf(innovation -> innovation.getInnovation() > maxInnovation); // RemoveIf car sinon en bouclant pour retirer lançait une java.util.ConcurrentModificationException

		}

		return rslt;
	}

	// Vérifier si une mutation possédant le même n° d'innovation n'est pas déjà survenue pour les 2 parents
	private int checkIncrementInnovation(ANN genome1, ANN genome2) {
		int inno = getGlobalInnovation();
		boolean incrInno = true;

		for (Innovation innovation : genome1.Innovations) {
			if (innovation.innovation == inno) { // Une mutation identique est déjà survenue, le n° d'innovation ne sera pas incrémenté
				incrInno = false;
				break;
			}
		}

		for (Innovation innovation : genome2.Innovations) {
			if (innovation.innovation == inno) { // Une mutation identique est déjà survenue, le n° d'innovation ne sera pas incrémenté
				incrInno = false;
				break;
			}
		}

		if (incrInno) {
			globalInnovation++;
		}

		return globalInnovation;
	}

	// Va vérifier si une innovation identique n'existe pas déjà dans le génome (pour ne pas avoir de gènes en doublons)
	private boolean checkConnectionGeneIdenticalExists(ArrayList<ConnectionGene> connectionGenes, ConnectionGene connectionGene) {
		boolean ret = connectionGenes.contains(connectionGene); // Si la même innovation avec les mêmes n°, inputs et outputs

		if (!ret) { // Pas identiques
			for (ConnectionGene connection : connectionGenes) {
				if (connection.getInnovation() == connectionGene.getInnovation()) {
					//System.out.println("connection n° : " + connection + " == connectionGene n° : " + connectionGene);
					ret = true;
				}
				else if (connection.getInto() == connectionGene.getInto() && connection.getOut() == connectionGene.getOut()) {
					//System.out.println("connection inputs/outputs : " + connection + " == connectionGene inputs/outputs : " + connectionGene);
					ret = true;
				}
			}
		}

		return !ret;
	}

	// Obtenir l'ANN comprenant le plus de ConnectionGene
	public int getLargerANN() {
		int max = 0;

		for (ANN ann : ANNs) {
			if (ann.conGeneList.size() > max) {
				max = ann.conGeneList.size();
			}
		}

		return max;
	}

	// Obtenir le premier membre de l'espèce
	public ANN getFirstANN() {
		for (ANN ann : ANNs) {
			if (ann.first) {
				return ann;
			}
		}

		return null; // Le premier membre de l'espèce ne se fera jamais remove
	}

	// Générer des enfants (ANN) à partir des ANNs restants
	public ArrayList<ANN> crossANNs(int numChildren) {
		// Faire des enfants selon les meilleurs fits
		Random rand = new Random();
		ArrayList<ANN> children = new ArrayList<>();

		if (ANNs.size() == 0) {
			return children;
		}

		for (int i = 0; i < numChildren; i++) {

			// On prend 2 génomes au hasard (on peut très bien prendre 2x le même enfant (mitose on va dire))
			int rand1 = rand.nextInt(ANNs.size());
			int rand2 = rand.nextInt(ANNs.size());

			ANN genome1 = ANNs.get(rand1);
			ANN genome2 = ANNs.get(rand2);

			// Recette pour faire l'enfant :
			// - Pour les gènes ayant le même n° d'innovation (communs aux 2 parents), on tire le gène à garder au hasard parmi les 2 parents
			// - Pour les gènes n'ayant pas le même n° d'innovation (disjoints ou excessifs) :
			//    --> Si les fits des parents sont identiques, on tire le gène à garder au hasard parmi les 2 parents
			//    --> Sinon on garde le gène du parent ayant le meilleur fit

			// Gène disjoint : gène n'ayant pas le même n° d'innovation du parent, mais se trouvant dans le range des n° d'innovations du parent
			// Gène excessif : gène n'ayant pas le même n° d'innovation du parent, et se trouvant en dehors du range des n° d'innovations du parent

			ArrayList<ConnectionGene[]> genomesCommuns = commonsElements(genome1.conGeneList, genome2.conGeneList);
			ArrayList<ConnectionGene> genomesDisjointsParent1 = disjointsElements(genome2.conGeneList, genome1.conGeneList);
			ArrayList<ConnectionGene> genomesDisjointsParent2 = disjointsElements(genome1.conGeneList, genome2.conGeneList);
			ArrayList<ConnectionGene> genomesExcessifsParent1 = excessElements(genome2.conGeneList, genome1.conGeneList);
			ArrayList<ConnectionGene> genomesExcessifsParent2 = excessElements(genome1.conGeneList, genome2.conGeneList);

			ArrayList<ConnectionGene> genomeChild = new ArrayList<>();

			// Pour les gènes ayant le même n° d'innovation (communs aux 2 parents), on tire le gène à garder au hasard parmi les 2 parents
			for (ConnectionGene[] connectionGenes : genomesCommuns) {
				if (rand.nextInt(100) < 50) { // Tirer au hasard le gène entre les 2 parents
					if (checkConnectionGeneIdenticalExists(genomeChild, connectionGenes[0])) { // N'ajouter la connection que s'il n'existe pas déjà dans le génome
						genomeChild.add(connectionGenes[0]); // Parent 1
					}
				}
				else {
					if (checkConnectionGeneIdenticalExists(genomeChild, connectionGenes[1])) {
						genomeChild.add(connectionGenes[1]); // Parent 2
					}
				}
			}

			// Pour les gènes n'ayant pas le même n° d'innovation (disjoints ou excessifs)
			double fit1 = genome1.fit;
			double fit2 = genome2.fit;

			// Si les fits des parents ne sont pas identiques, on garde le gène du parent ayant le meilleur fit
			if (fit1 > fit2) { // Récupérer les gènes du parent 1
				for (ConnectionGene connection : genomesDisjointsParent1) {
					if (checkConnectionGeneIdenticalExists(genomeChild, connection)) {
						genomeChild.add(connection);
					}
				}

				for (ConnectionGene connection : genomesExcessifsParent1) {
					if (checkConnectionGeneIdenticalExists(genomeChild, connection)) {
						genomeChild.add(connection);
					}
				}
			}
			else if (fit1 < fit2) { // Récupérer les gènes du parent 2
				for (ConnectionGene connection : genomesDisjointsParent2) {
					if (checkConnectionGeneIdenticalExists(genomeChild, connection)) {
						genomeChild.add(connection);
					}
				}

				for (ConnectionGene connection : genomesExcessifsParent2) {
					if (checkConnectionGeneIdenticalExists(genomeChild, connection)) {
						genomeChild.add(connection);
					}
				}
			}
			else {
				// Si les fits des parents sont identiques, on tire le gène à garder au hasard parmi les 2 parents
				if (rand.nextInt(100) < 50) { // Tirer au hasard le gène entre les 2 parents
					for (ConnectionGene connection : genomesDisjointsParent1) {
						if (checkConnectionGeneIdenticalExists(genomeChild, connection)) {
							genomeChild.add(connection);
						}
					}

					for (ConnectionGene connection : genomesExcessifsParent1) {
						if (checkConnectionGeneIdenticalExists(genomeChild, connection)) {
							genomeChild.add(connection);
						}
					}
				}
				else {
					for (ConnectionGene connection : genomesDisjointsParent2) {
						if (checkConnectionGeneIdenticalExists(genomeChild, connection)) {
							genomeChild.add(connection);
						}
					}

					for (ConnectionGene connection : genomesExcessifsParent2) {
						if (checkConnectionGeneIdenticalExists(genomeChild, connection)) {
							genomeChild.add(connection);
						}
					}
				}
			}

			// On le fait muter (ou pas car 20% de chance de mutation)
			if (rand.nextInt(100) < tauxMutation) {
				int gene = -1;
				float weight;
				int inno;
				int nbHiddenNodes;
				int maxGeneConnection;
				int maxGeneDestination;
				int numGeneConnection;
				int numGeneDestination;

				switch (rand.nextInt(4)) {
					case 0: // Ajout d'une nouvelle connexion

						// Prendre un noeud au hasard et rajouter une connexion avec un autre noeud au hasard (pas le même, ni un avec lequel il est déjà connecté, ni vers une entrée)

						nbHiddenNodes = 0;

						if (genome1.getHiddensIndexs().size() > 0) {
							nbHiddenNodes = Collections.max(genome1.getHiddensIndexs());
						}

						maxGeneConnection = genome1.getNbInputs() + 1 + nbHiddenNodes; // +1 pour le biais
						maxGeneDestination = genome1.getNbInputs() + 1 + nbHiddenNodes + genome1.getNbOutputs();

						do { // Boucler tant qu'on ne se relie pas à soi-même
							numGeneConnection = rand.nextInt(maxGeneConnection);

							// Pour avoir un range de valeurs, ne pas prendre les entrées mais garder les cachés
							numGeneDestination = rand.nextInt(maxGeneDestination - genome1.getNbInputs()) + genome1.getNbInputs();

							// Si on a pris une sortie, on remplace la valeur par le numéro du node de sortie
							if (numGeneDestination >= maxGeneDestination - genome1.getNbOutputs()) {
								numGeneDestination = genome1.getOutputs().get(maxGeneDestination - numGeneDestination - 1).getNumero();
							}

							// Check pour voir si la connexion n'existe pas déjà
							for (ConnectionGene connectionGene : genomeChild) {
								if (connectionGene.getInto() == numGeneConnection && connectionGene.getOut() == numGeneDestination || connectionGene.getInto() == numGeneDestination && connectionGene.getOut() == numGeneConnection) {
									// Si la connexion existe déjà, on ne la prend pas
									numGeneDestination = numGeneConnection;
									break;
								}
							}
						} while (numGeneConnection == numGeneDestination);

						// Récupérer le n° d'innovation global et vérifier dans la liste dse Innovations si une mutation identique s'est déjà produite, dans ce cas-là, ne pas incrémenter le n° d'innovation
						inno = checkIncrementInnovation(genome1, genome2);

						// nextFloat donne une valeur entre 0 et 1 du coup * 10 pour avoir une valeur en 0 et 10
						weight = rand.nextFloat() * 10;

						if (rand.nextInt(100) < 50) { // Une chance sur deux qu'il soit en négatif valeur entre -10 et 10
							weight -= weight * 2;
						}

						ConnectionGene newConnection = new ConnectionGene(numGeneConnection, numGeneDestination, inno, weight, true);
						genomeChild.add(newConnection);

						break;
					case 1: // Ajout d'un nouveau noeud

						// On prend 2 noeuds au hasard, et on rajoute un nouveau noeud entre son entrée et sa sortie, puis s'ils avaient déjà une connexion entre eux, on désactive l'ancienne connexion

						nbHiddenNodes = 0;

						if (genome1.getHiddensIndexs().size() > 0) {
							nbHiddenNodes = Collections.max(genome1.getHiddensIndexs());
						}

						maxGeneConnection = genome1.getNbInputs() + 1 + nbHiddenNodes; // +1 pour le biais
						maxGeneDestination = genome1.getNbInputs() + 1 + nbHiddenNodes + genome1.getNbOutputs();

						boolean connexionExisteDeja;

						do { // Boucler tant qu'on ne se relie pas à soi-même
							connexionExisteDeja = false;

							numGeneConnection = rand.nextInt(maxGeneConnection);

							// Pour avoir un range de valeurs, ne pas prendre les entrées mais garder les cachés
							numGeneDestination = rand.nextInt(maxGeneDestination - genome1.getNbInputs()) + genome1.getNbInputs();

							// Si on a pris une sortie, on remplace la valeur par le numéro du node de sortie
							if (numGeneDestination >= maxGeneDestination - genome1.getNbOutputs()) {
								numGeneDestination = genome1.getOutputs().get(maxGeneDestination - numGeneDestination - 1).getNumero();
							}

							// Check pour voir si la connexion n'existe pas déjà
							for (int j = 0; j < genomeChild.size(); j++) {
								ConnectionGene connectionGene = genomeChild.get(j);
								if (connectionGene.getInto() == numGeneConnection && connectionGene.getOut() == numGeneDestination || connectionGene.getInto() == numGeneDestination && connectionGene.getOut() == numGeneConnection) {
									// Si la connexion existe déjà, on ne la prend pas
									connexionExisteDeja = true;
									gene = j;
									break;
								}
							}
						} while (numGeneConnection == numGeneDestination);

						// On récupère un nouveau n° de noeud (nombre de noeuds cachés + nombre d'entrées + biais + 1)
						int max = genome1.getHiddens().size() + genome1.getNbInputs() + 2;

						// Récupérer le n° d'innovation global et vérifier dans la liste dse Innovations si une mutation identique s'est déjà produite, dans ce cas-là, ne pas incrémenter le n° d'innovation
						inno = checkIncrementInnovation(genome1, genome2);

						// nextFloat donne une valeur entre 0 et 1 du coup * 10 pour avoir une valeur en 0 et 10
						weight = rand.nextFloat() * 10;

						if (rand.nextInt(100) < 50) { // Une chance sur deux qu'il soit en négatif valeur entre -10 et 10
							weight -= weight * 2;
						}

						ConnectionGene newConnection1 = new ConnectionGene(numGeneConnection, max, inno, weight, true);

						// Récupérer le n° d'innovation global et vérifier dans la liste dse Innovations si une mutation identique s'est déjà produite, dans ce cas-là, ne pas incrémenter le n° d'innovation
						inno = checkIncrementInnovation(genome1, genome2);

						// nextFloat donne une valeur entre 0 et 1 du coup * 10 pour avoir une valeur en 0 et 10
						weight = rand.nextFloat() * 10;

						if (rand.nextInt(100) < 50) { // Une chance sur deux qu'il soit en négatif valeur entre -10 et 10
							weight -= weight * 2;
						}

						ConnectionGene newConnection2 = new ConnectionGene(max, numGeneDestination, inno, weight, true);

						// Désactiver le gène qui était là avant
						if (connexionExisteDeja) {
							genomeChild.get(gene).setEnabled(false);
						}

						genomeChild.add(newConnection1);
						genomeChild.add(newConnection2);

						break;
					case 2: // Changer le poids d'une connexion

						// On fait comme dans l'exemple génétique du prof (avec size) : 1 chance sur 2 de totalement changer la valeur, ou augmenter/diminuer la valeur d'une certaine valeur flottante aléatoire entre 1 et 5

						if (genomeChild.size() > 0) {
							gene = rand.nextInt(genomeChild.size());

							if (rand.nextInt(2) == 0) {
								// nextFloat donne une valeur entre 0 et 1 du coup * 10 pour avoir une valeur en 0 et 10
								weight = rand.nextFloat() * 10;

								if (rand.nextInt(100) < 50) { // Une chance sur deux qu'il soit en négatif valeur entre -10 et 10
									weight -= weight * 2;
								}
							}
							else {
								weight = genomeChild.get(gene).getWeight();

								if (rand.nextInt(2) == 0) {
									weight += rand.nextFloat() * rand.nextInt(5); // On incrémente d'une valeur flottante aléatoire entre 0 et 5

									if (weight > 10) {
										weight = 10;
									}
								}
								else {
									weight -= rand.nextFloat() * rand.nextInt(5); // On décrémente d'une valeur flottante aléatoire entre 0 et 5

									if (weight < -10) {
										weight = -10;
									}
								}
							}

							genomeChild.get(gene).setWeight(weight);
						}

						break;
					case 3: // Activer ou désactiver une connexion
						if (genomeChild.size() > 0) {
							gene = rand.nextInt(genomeChild.size());
							boolean enable = genomeChild.get(gene).switchEnabled();

						}

						break;
				}
			}


			// On peut donc créer l'enfant
			ANN child = new ANN(genomeChild, genome1.getNbInputs(), genome1.getNbOutputs(), genome1.getNbMaxHiddenNodes());
			children.add(child);
		}

		return children;
	}

	@Override
	public String toString() {
		return "(Espèce " + specieNumber + ", stable : " + stableCounter + ", n° innovation global : " + globalInnovation + ", nombre d'individus : " + ANNs.size() + ")";
	}
}
