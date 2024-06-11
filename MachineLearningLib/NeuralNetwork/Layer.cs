using MachineLearningLib.ActivationFunctions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningLib.NeuralNetwork
{
    public class Layer
    {
        public float[] Biases;
        public float[] NeuronsSum;
        public float[] NeuronsAF;
        public float[] Errors;
        public float[][] Weights;

        public Layer FollowingLayer;
        public Layer PreviousLayer;

        public IActivationFunction ActivationFunction = new Sigmoid();

        public object Tag;

        public Layer(int neurons)
        {
            Biases = new float[neurons];
            NeuronsSum = new float[neurons];
            NeuronsAF = new float[neurons];
            Errors = new float[neurons];
        }

        public virtual void InitFromPreviousLayer()
        {
            Weights = new float[NeuronsSum.Length][];
            Random r = new Random();
            for(int i = 0; i < NeuronsSum.Length; i++)
            {
                Weights[i] = new float[PreviousLayer.NeuronsSum.Length];
                Biases[i] = (float)r.NextDouble() * 2 - 1;
                for (int j = 0; j < PreviousLayer.NeuronsSum.Length; j++)
                {
                    Weights[i][j] = (float)r.NextDouble() * 2 - 1;
                }
            }
        }

        public virtual void Calculate()
        {
            for(int i = 0; i < NeuronsSum.Length; i++)
            {
                NeuronsSum[i] = 0;
                for(int j = 0; j < PreviousLayer.NeuronsSum.Length; j++)
                {
                    NeuronsSum[i] += PreviousLayer.NeuronsAF[j] * Weights[i][j];
                }
                NeuronsSum[i] += Biases[i];
                NeuronsAF[i] = ActivationFunction.Evaluate(NeuronsSum[i]);
            }
            FollowingLayer.Calculate();
        }

        public virtual void Train(float learningRate)
        {
            for(int i = 0; i < NeuronsSum.Length; i++)
            {
                Errors[i] = 0;
                for(int j = 0; j < FollowingLayer.NeuronsSum.Length; j++)
                {
                    Errors[i] += FollowingLayer.Errors[j] * FollowingLayer.Weights[j][i];
                }
                Errors[i] *= ActivationFunction.Derivative(NeuronsSum[i]);
                for(int j = 0; j < PreviousLayer.NeuronsSum.Length; j++)
                {
                    Weights[i][j] += learningRate * Errors[i] * PreviousLayer.NeuronsAF[j];
                }
                Biases[i] += learningRate * Errors[i];
            }
            PreviousLayer.Train(learningRate);
        }
    }
}
