package neuralnet.network;

import java.util.Vector;

public class Neuron {

    private static double eta = 0.15; // 0 to 1, overall net training rate
    private static double alpha = 0.5; // 0 to n, multiplier of last weight change
    private double outputValue;
    private double myGradient;
    private int myIndex;

    private Vector<Connection> outputWeights;

    public Neuron(int numOutputs, int myIndex) {
        outputWeights = new Vector<Connection>();
        for (int i = 0; i < numOutputs; i++) {
            outputWeights.add(new Connection());
            outputWeights.lastElement().weight = randomWeight();
        }
        this.myIndex = myIndex;
    }

    public void setOutputValue(double value) {
        this.outputValue = value;
    }

    public double getOutputValue() {
        return outputValue;
    }

    public void feedForward(Layer previousLayer) {
        double sum = 0.0;

        // sum the previous layers' outputs (which are our inputs(
        // include the bias node from the previous layer
        for (int i = 0; i < previousLayer.size(); i++) {
            sum += previousLayer.get(i).getOutputValue()
                    * previousLayer.get(i).outputWeights.get(myIndex).weight;
        }
        outputValue = Neuron.transferFunction(sum);
    }

    public void calculateOutputGradients(double targetValue) {
        double delta = targetValue - outputValue;
        myGradient = delta * Neuron.transferFunctionDerivative(outputValue);
    }

    public void calculateHiddenGradients(Layer nextLayer) {
        double dow = sumDOW(nextLayer);
        myGradient = dow * Neuron.transferFunctionDerivative(outputValue);

    }

    public void updateInputWeights(Layer previousLayer) {
        // The weights to be updated are in the Connection class
        // in the neurons in the preceding layer
        for (int i = 0; i < previousLayer.size(); i++) {
            Neuron currentNeuron = previousLayer.elementAt(i);
            double oldDeltaWeight = currentNeuron.outputWeights.get(myIndex).deltaWeight;

            double newDeltaWeight
                    = eta
                    * currentNeuron.getOutputValue()
                    * myGradient
                    + alpha
                    * oldDeltaWeight;

            currentNeuron.outputWeights.get(myIndex).deltaWeight = newDeltaWeight;
            currentNeuron.outputWeights.get(myIndex).weight += newDeltaWeight;
        }
    }

    private static double transferFunction(double value) {
        return Math.tanh(value);
    }

    private static double transferFunctionDerivative(double value) {
        return 1.0 - value * value;
    }

    private static double randomWeight() {
        return Math.random();
    }

    private double sumDOW(Layer nextLayer) {
        double sum = 0.0;

        // sum the contributions of the errors at the nodes we feed
        for (int i = 0; i < nextLayer.size() - 1; i++) {
            sum += outputWeights.get(i).weight * nextLayer.get(i).myGradient;
        }
        return sum;
    }

}
