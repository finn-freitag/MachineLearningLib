using MachineLearningLib.NeuralNetwork;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningLib.Optimizers
{
    public class AdaGradOptimizer : IOptimizer
    {
        public void Train(Layer layer, int neuronIndex, float[][] weights, float[] biases, float[] errorsWithAFDerivative, float[] inputs, float learningRate)
        {
            float[][] WeightAccumulators = null;
            float[] BiasAccumulators = null;

            if(layer.OptimizerData == null)
            {
                WeightAccumulators = new float[weights.Length][];
                BiasAccumulators = new float[biases.Length];
                for (int i = 0; i < weights.Length; i++)
                {
                    WeightAccumulators[i] = new float[weights[i].Length];
                }
                layer.OptimizerData = new object[] { WeightAccumulators, BiasAccumulators };
            }
            else
            {
                WeightAccumulators = ((object[])layer.OptimizerData)[0] as float[][];
                BiasAccumulators = ((object[])layer.OptimizerData)[1] as float[];
            }

            float epsilon = 1e-8f;
            for (int j = 0; j < weights[neuronIndex].Length; j++)
            {
                WeightAccumulators[neuronIndex][j] += errorsWithAFDerivative[neuronIndex] * errorsWithAFDerivative[neuronIndex];

                float adjustedLearningRate = learningRate / (float)Math.Sqrt(WeightAccumulators[neuronIndex][j] + epsilon);
                weights[neuronIndex][j] += adjustedLearningRate * errorsWithAFDerivative[neuronIndex] * inputs[j];
            }

            BiasAccumulators[neuronIndex] += errorsWithAFDerivative[neuronIndex] * errorsWithAFDerivative[neuronIndex];

            float adjustedBiasLearningRate = learningRate / (float)Math.Sqrt(BiasAccumulators[neuronIndex] + epsilon);
            biases[neuronIndex] += adjustedBiasLearningRate * errorsWithAFDerivative[neuronIndex];
        }
    }
}
