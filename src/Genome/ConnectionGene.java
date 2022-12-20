package Genome;

// Classe étant la connexion entre des noeuds, contenant son entrée et sortie, le n° d'innovation, son poids et s'il est activé ou non. Est un gène du génome avec NodeGene.

/*
 * Classe étant la connexion entre des noeuds, contenant son entrée et sortie, le
 * n° d'innovation, son poids et s'il est activé ou non. Est un gène du génome
 * avec NodeGene.
 */
public class ConnectionGene {
	private final int into;
	private final int out;
	private final int innovation;
	private float weight;
	private boolean enabled;

	/*
	 * Constructeur de la classe ConnectionGene, prend en paramètre l'entrée, la
	 * sortie, le n° d'innovation, le poids et si elle est activée ou non.
	 *
	 */
	public ConnectionGene(int into, int out, int innovation, float weight, boolean enabled) {
		this.into = into;
		this.out = out;
		this.innovation = innovation;
		this.weight = weight;
		this.enabled = enabled;
	}

	public int getInto() {
		return into;
	}

	public int getOut() {
		return out;
	}

	public int getInnovation() {
		return innovation;
	}

	public float getWeight() {
		return weight;
	}

	public void setWeight(float weight) {
		this.weight = weight;
	}

	public boolean isEnabled() {
		return enabled;
	}

	public boolean switchEnabled() {
		enabled = !enabled;
		return enabled;
	}

	public void setEnabled(boolean enabled) {
		this.enabled = enabled;
	}

	@Override
	public String toString() {
		return "[" + into + ", " + out + ", " + innovation + ", " + weight + ", " + enabled + "]";
	}
}
