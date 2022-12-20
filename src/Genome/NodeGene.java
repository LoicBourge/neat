package Genome;

import java.util.ArrayList;

/*
*Cette classe représente un gène de noeud qui contient son n° d'innovation, son type (input, output, hidden) et son numéro de layer.
 */
public class NodeGene {
	public enum NodeType {
		Input,
		Output,
		Hidden
	}

	private float value;
	private final NodeType type;
	private final int numero;

	private final ArrayList<ConnectionGene> incomingCon = new ArrayList<>();

	public NodeGene(float value, NodeType type, int numero) {
		super();
		this.value = value;
		this.type = type;
		this.numero = numero;
	}

	public float getValue() {
		return value;
	}

	public void setValue(float value) {
		this.value = value;
	}

	public NodeType getType() {
		return type;
	}

	public int getNumero() {
		return numero;
	}

	public ArrayList<ConnectionGene> getIncomingCon() {
		return incomingCon;
	}

	@Override
	public String toString() {
		return "(" + numero + " = " + type + " " + value + " connections : " + incomingCon.size() + ")";
	}
}
