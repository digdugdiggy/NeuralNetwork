package neuralnet.network;

import java.util.Vector;

public class Net {

    // number of training samples to average over
    private static double recentAverageSmoothingFactor = 100.0;
    private Vector<Layer> layers;
    private double error;
    private double recentAverageError;

    public Net(Vector<Integer> topology) {
        layers = new Vector<Layer>();
        int numLayers = topology.size();
        for (int layerNum = 0; layerNum < numLayers; layerNum++) {
            layers.add(new Layer());
            int numOutputs = layerNum == topology.size() - 1 ? 0 : topology.get(layerNum + 1);

            // we have a new layer, now fill it with neurons and add a bias neuron in each layer
            for (int neuronNum = 0; neuronNum <= topology.get(layerNum); neuronNum++) {
                layers.lastElement().add(new Neuron(numOutputs, neuronNum));
                System.out.println("Made a neuron");
            }
            // force the bias nodes output to 1.0
            layers.lastElement().lastElement().setOutputValue(1.0);
        }
    }

    public void feedForward(Vector<Integer> inputValues) {
        // assign the input values into the input neuron

        for (int i = 0; i < inputValues.size(); i++) {
            layers.get(0).get(i).setOutputValue(inputValues.get(i));
        }

        // forward propigation
        for (int layerNum = 1; layerNum < layers.size(); layerNum++) {
            Layer previousLayer = layers.get(layerNum - 1);
            for (int i = 0; i < layers.get(layerNum).size() - 1; i++) {
                layers.get(layerNum).get(i).feedForward(previousLayer);
            }
        }
    }

    public void backProp(Vector<Integer> targetValues) {
        // calculate overall net error RMS

        Layer outputLayer = layers.lastElement();
        error = 0.0;

        for (int i = 0; i < outputLayer.size() - 1; i++) {
            double delta = targetValues.get(i) - outputLayer.get(i).getOutputValue();
            error += delta * delta;
        }
        error /= outputLayer.size() - 1;    // get average error 
        error = Math.sqrt(error);   // RMS

        // implement a recent average measurement
        recentAverageError
                = (recentAverageError * recentAverageSmoothingFactor + error)
                / (recentAverageSmoothingFactor + 1.0);

        // calculate output Layer gradients        
        for (int i = 0; i < outputLayer.size() - 1; i++) {
            outputLayer.get(i).calculateOutputGradients(targetValues.get(i));
        }

        // calculate hidden layer gradients
        for (int layerNum = layers.size() - 2; layerNum > 0; layerNum--) {
            Layer hiddenLayer = layers.get(layerNum);
            Layer nextLayer = layers.get(layerNum + 1);

            for (int i = 0; i < hiddenLayer.size(); i++) {
                hiddenLayer.get(i).calculateHiddenGradients(nextLayer);
            }
        }
        // for all layers from outputs to first hidden layer.
        // update connection weights

        for (int layerNum = layers.size() - 1; layerNum > 0; layerNum--) {
            Layer layer = layers.get(layerNum);
            Layer previousLayer = layers.get(layerNum - 1);

            for (int i = 0; i < layer.size() - 1; i++) {
                layer.get(i).updateInputWeights(previousLayer);
            }
        }
    }

    public void getResults(Vector<Double> resultValues) {
        resultValues.clear();
        for (int i = 0; i < layers.lastElement().size() - 1; i++) {
            resultValues.add(layers.lastElement().get(i).getOutputValue());
        }
    }

    public double getRecentAverageError() {
        return recentAverageError;
    }

}
