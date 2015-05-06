// Neural Net program
// by Charles Fahselt

// Ported from C++ to Java
// Original code by Dave Miller at http://www.millermattson.com/dave/?p=54

// this program is intended to learn about neural networks for use in
// game AI eventually.

package neuralnet;

import neuralnet.network.Net;
import java.util.Vector;

public class Main {
    
    // This is the main driver class.
    // It models a neural net "AND gate"

    public static void main(String[] args) {

        // create test data and expected outputs to train the net
        TrainingData data = new TrainingData(2000);
        
        // Topology holds the overal structure of nodes in the net.
        // 2, 4, 1 means 2 inputs, 4 internal nodes, and one output
        Vector<Integer> topology = new Vector<Integer>();
        // number of nodes in each tier. There is also another bias node per tier.
        topology.add(2); // 2 in first tier means two inputs
        topology.add(4); // 4 inner neurons
        topology.add(1); // 1 output neuron
        Net myNeuralNet = new Net(topology);

        // init all the variables
        Vector<Integer> inputValues, targetValues;
        Vector<Double>  resultValues;
        inputValues = new Vector<Integer>();
        targetValues = new Vector<Integer>();
        resultValues = new Vector<Double>();
        int trainingPass = 0;

        while (!data.isEof(trainingPass)) {
            
            System.out.println("\nPass #: " + trainingPass);

            // get new input data and feed it forward
            inputValues = data.getNextInputs(trainingPass);
            if (inputValues.size() != topology.get(0)) {
                break;
            }
            showVectorValues("Inputs:", inputValues);
            myNeuralNet.feedForward(inputValues);

            // collect the results from the net
            myNeuralNet.getResults(resultValues);
            showVectorValuesD("Outputs:", resultValues);

            // train the net with what the result should have been
            targetValues = data.getTargetOutputs(trainingPass);
            showVectorValues("Targets:", targetValues);

            myNeuralNet.backProp(targetValues);

            // report how well the training is going
            System.out.println("Net recent average error: " + myNeuralNet.getRecentAverageError());
            
            ++trainingPass;

        }
        System.out.println("done");
    }

    public static void showVectorValues(String label, Vector<Integer> v) {
        System.out.print(label + " ");
        for (int i = 0; i < v.size(); i++) {
            System.out.print(v.get(i) + ", ");
        }
        System.out.printf("\n");

    }
    public static void showVectorValuesD(String label, Vector<Double> v) {
        System.out.print(label + " ");
        for (int i = 0; i < v.size(); i++) {
            System.out.printf(v.get(i) + " \n");
        }

    }

}
