package neuralnet;

import java.util.Vector;

// Class to provide input data to the neural net for training
// This version produces two inputs, one output, and emulates an AND logic gate

class TrainingData {
    
    private int NUM_RUNS;
    
    private Vector<Integer> input1;
    private Vector<Integer> input2;

    private Vector<Integer> intendedOutput;   

    TrainingData(int numberOfRuns) {
        NUM_RUNS = numberOfRuns;
        // initialize the Vector data members
        input1 = new Vector<Integer>();
        input2 = new Vector<Integer>();
        intendedOutput = new Vector<Integer>();
        
        for (int i = 0; i < NUM_RUNS; i++) {
            // randomly pick zero or one.
            input1.add((Math.random() < 0.5) ? 0 : 1);
            input2.add((Math.random() < 0.5) ? 0 : 1);

            // Intended output is the two inputs AND gated together
            if (input1.lastElement() == 1 && input2.lastElement() == 1) {
                intendedOutput.add(1);
            } else {
                intendedOutput.add(0);
            }
        }
    }

    public Vector<Integer> getTargetOutputs(int index) {
        // return a Vector with all of the intended output values
        Vector<Integer> targetOutputs = new Vector<Integer>();        
        targetOutputs.add(intendedOutput.get(index));        
        return targetOutputs;
    }

    public Vector<Integer> getNextInputs(int index) {
        // returns a vector with all of the input values
        Vector<Integer> inputValues = new Vector<Integer>();        
        inputValues.add(input1.get(index));
        inputValues.add(input2.get(index));        
        return inputValues;
    }

    public boolean isEof(int index) {
        // checks for end of file
        if (index >= NUM_RUNS) {
            return true;
        } else {
            return false;
        }
    }

}
