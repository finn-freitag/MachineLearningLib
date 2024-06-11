using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningLib.NeuralNetwork
{
    public class OutputLayer : Layer
    {
        protected float[] desiredOutputs;

        public OutputLayer(int neurons) : base(neurons)
        {
        }

        public void SetDesiredOutputs(float[] desiredOutputs)
        {
            if (desiredOutputs.Length != NeuronsSum.Length)
                throw new InvalidOperationException("Wrong amount of desired outputs!");
            this.desiredOutputs = desiredOutputs;
        }

        public float[] GetOutputs()
        {
            return NeuronsAF;
        }

        public override void Calculate()
        {
            for (int i = 0; i < NeuronsSum.Length; i++)
            {
                NeuronsSum[i] = 0;
                for (int j = 0; j < PreviousLayer.NeuronsSum.Length; j++)
                {
                    NeuronsSum[i] += PreviousLayer.NeuronsAF[j] * Weights[i][j];
                }
                NeuronsSum[i] += Biases[i];
                NeuronsAF[i] = ActivationFunction.Evaluate(NeuronsSum[i]);
            }
        }

        public override void Train(float learningRate)
        {
            for (int i = 0; i < NeuronsSum.Length; i++)
            {
                Errors[i] = (desiredOutputs[i] - NeuronsAF[i]) * ActivationFunction.Derivative(NeuronsSum[i]);
                for (int j = 0; j < PreviousLayer.NeuronsSum.Length; j++)
                {
                    Weights[i][j] += learningRate * Errors[i] * PreviousLayer.NeuronsAF[j];
                }
                Biases[i] += learningRate * Errors[i];
            }
            PreviousLayer.Train(learningRate);
        }
    }
}
