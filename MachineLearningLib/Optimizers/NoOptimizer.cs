using MachineLearningLib.Accelerators;
using MachineLearningLib.NeuralNetwork;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningLib.Optimizers
{
    public class NoOptimizer : IOptimizer
    {
        public void Train(Layer layer, int neuronIndex, float[][] weights, float[] biases, float[] errorsWithAFDerivative, float[] inputs, float learningRate)
        {
            weights[neuronIndex] = layer.Accelerator.Add(weights[neuronIndex], layer.Accelerator.Multiply(learningRate * errorsWithAFDerivative[neuronIndex], inputs));
            biases[neuronIndex] += learningRate * errorsWithAFDerivative[neuronIndex];
        }
    }
}
