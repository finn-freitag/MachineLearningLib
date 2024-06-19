using MachineLearningLib.ActivationFunctions;
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
            var sigmoid = new SigmoidActivation();
            for (int i = 0; i < NeuronsSum.Length; i++)
            {
                float sum = 0f;
                for (int j = 0; j < PreviousLayer.NeuronsSum.Length; j++)
                {
                    sum += Weights[i][j] * PreviousLayer.NeuronsAF[j];
                }
                sum += Biases[i];
                NeuronsSum[i] = sum;
                NeuronsAF[i] = sigmoid.Evaluate(sum);
            }
        }

        public override void Train(float learningRate)
        {
            var sigmoid = new SigmoidActivation();
            for (int i = 0; i < NeuronsSum.Length; i++)
            {
                Errors[i] = (desiredOutputs[i] - NeuronsAF[i]) * sigmoid.Derivative(NeuronsSum[i]);
                for (int j = 0; j < PreviousLayer.NeuronsAF.Length; j++)
                {
                    Weights[i][j] += learningRate * Errors[i] * PreviousLayer.NeuronsAF[j];
                }
                Biases[i] += learningRate * Errors[i];
            }
            PreviousLayer.Train(learningRate);
        }
    }
}
