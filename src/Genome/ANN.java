package Genome;

import java.util.ArrayList;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;

/*
 * Classe contenant un réseau de neurones (ANN = A Neural Network), c'est le
 * génome. Va générer le réseau et l'évaluer.
 */

public class ANN implements Comparable<ANN> {
    private final int nbMaxHiddenNodes;
    private final int nbInputs;
    private final int nbOutputs;
    private final SortedMap<Integer, NodeGene> nodes = new TreeMap<>();

    public ArrayList<ConnectionGene> conGeneList;
    public ArrayList<Innovation> Innovations;
    public float[] outputs;
    public double fit;
    public boolean first; // Si c'est le premier génome de la génération

    /*
     * Constructeur de la classe ANN, prend en paramètre le nombre de noeuds cachés
     * maximum, le nombre d'entrées et le nombre de sorties. Va générer le réseau de
     * neurones.
     */
    public ANN(ArrayList<ConnectionGene> con, int in, int out, int max) {
        conGeneList = new ArrayList<>(con);
        nbInputs = in;
        nbOutputs = out;
        nbMaxHiddenNodes = max;
        fit = 0;
        first = false;

        generateInnovations();
        generateNetwork();
    }

    public int getNbInputs() {
        return nbInputs;
    }

    public int getNbOutputs() {
        return nbOutputs;
    }

    public int getNbMaxHiddenNodes() {
        return nbMaxHiddenNodes;
    }

    // Retourne les noeuds d'entrées
    public ArrayList<NodeGene> getInputs() {
        ArrayList<NodeGene> rslt = new ArrayList<>();

        for (Map.Entry<Integer, NodeGene> mapEntry : nodes.entrySet()) {
            if (mapEntry.getValue().getType() == NodeGene.NodeType.Input) {
                rslt.add(mapEntry.getValue());
            }
        }

        return rslt;
    }

    // Retourne les noeuds cachés
    public ArrayList<NodeGene> getHiddens() {
        ArrayList<NodeGene> rslt = new ArrayList<>();

        for (Map.Entry<Integer, NodeGene> mapEntry : nodes.entrySet()) {
            if (mapEntry.getValue().getType() == NodeGene.NodeType.Hidden) {
                rslt.add(mapEntry.getValue());
            }
        }

        return rslt;
    }

    // Retourne les indexs des noeuds cachés
    public ArrayList<Integer> getHiddensIndexs() {
        ArrayList<Integer> rslt = new ArrayList<>();

        for (Map.Entry<Integer, NodeGene> mapEntry : nodes.entrySet()) {
            if (mapEntry.getValue().getType() == NodeGene.NodeType.Hidden) {
                rslt.add(mapEntry.getKey());
            }
        }

        return rslt;
    }

    // Retourne les noeuds de sorties
    public ArrayList<NodeGene> getOutputs() {
        ArrayList<NodeGene> rslt = new ArrayList<>();

        for (Map.Entry<Integer, NodeGene> mapEntry : nodes.entrySet()) {
            if (mapEntry.getValue().getType() == NodeGene.NodeType.Output) {
                rslt.add(mapEntry.getValue());
            }
        }

        return rslt;
    }

    // Générer la liste d'Innovations
    public void generateInnovations() {
        Innovations = new ArrayList<>();

        for (ConnectionGene connectionGene : conGeneList) {
            Innovations.add(new Innovation(connectionGene.getInnovation(), connectionGene.getInto(), connectionGene.getOut()));
        }

        Innovations.sort(Innovation::compareTo);
    }

    // Générer le réseau à partir des Connexions
    public void generateNetwork() {
        nodes.clear();

        // Input layer
        for (int i = 0; i < nbInputs; i++) {
            nodes.put(i, new NodeGene(0, NodeGene.NodeType.Input, i)); // Inputs
        }

        // Input supplémentaire biais, qui sera toujours à 1
        nodes.put(nbInputs, new NodeGene(1, NodeGene.NodeType.Input, nbInputs)); // Bias

        // Output layer
        for (int i = nbInputs + nbMaxHiddenNodes; i < nbInputs + nbMaxHiddenNodes + nbOutputs; i++) {
            nodes.put(i, new NodeGene(0, NodeGene.NodeType.Output, i));
        }

        // Hidden layer
        for (ConnectionGene con : conGeneList) {
            if (!nodes.containsKey(con.getInto())) {
                nodes.put(con.getInto(), new NodeGene(0, NodeGene.NodeType.Hidden, con.getInto()));
            }

            if (!nodes.containsKey(con.getOut())) {
                nodes.put(con.getOut(), new NodeGene(0, NodeGene.NodeType.Hidden, con.getOut()));
            }

            nodes.get(con.getOut()).getIncomingCon().add(con);
        }
    }

    // Évaluer le réseau une fois qu'il a fini de converger
    public float[] evaluateNetwork(float[] inputs) {
        float[] output = new float[nbOutputs];

        for (int i = 0; i < nbInputs; i++) {
            nodes.get(i).setValue(inputs[i]);
        }

        for (Map.Entry<Integer, NodeGene> mapEntry : nodes.entrySet()) {
            float sum = 0;
            int key = mapEntry.getKey();

            NodeGene node = mapEntry.getValue();

            if (key > nbInputs) {
                for (ConnectionGene conn : node.getIncomingCon()) {
                    if (conn.isEnabled()) {
                        sum += nodes.get(conn.getInto()).getValue() * conn.getWeight();
                    }
                }
                node.setValue(sigmoid(sum));
            }
        }

        for (int i = 0; i < nbOutputs; i++) {
            output[i] = nodes.get(nbInputs + nbMaxHiddenNodes + i).getValue();
        }

        outputs = output;
        return output;
    }

    // Fonction d'activation sigmoide, afin d'avoir le résultat entre 0 et 1
    private float sigmoid(float x) {
        return (float) (1 / (1 + Math.exp(-4.9 * x)));
    }

    @Override
    public int compareTo(ANN compare) {
        return (int) (compare.fit - this.fit); // Tri descendant
    }

    @Override
    public String toString() {
        return "(ANN Fit : " + fit + ", Nbre noeuds : " + nodes.size() + ", Nbre connexions : " + conGeneList.size() + " )";
    }
}
