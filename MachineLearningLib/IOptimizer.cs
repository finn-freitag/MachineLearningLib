using MachineLearningLib.NeuralNetwork;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningLib
{
    public interface IOptimizer
    {
        void Train(Layer layer, int neuronIndex, float[][] weights, float[] biases, float[] errorsWithAFDerivative, float[] inputs, float learningRate);
    }
}
