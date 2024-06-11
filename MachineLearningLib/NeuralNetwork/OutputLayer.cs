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
            Parallelizer(0, NeuronsSum.Length, (i) =>
            {
                NeuronsSum[i] = Accelerator.DotProduct(PreviousLayer.NeuronsAF, Weights[i]);
                NeuronsSum[i] += Biases[i];
                NeuronsAF[i] = ActivationFunction.Evaluate(NeuronsSum[i]);
            });
        }

        public override void Train(float learningRate)
        {
            Parallelizer(0, NeuronsSum.Length, (i) =>
            {
                Errors[i] = (desiredOutputs[i] - NeuronsAF[i]) * ActivationFunction.Derivative(NeuronsSum[i]);
                Weights[i] = Accelerator.Add(Weights[i], Accelerator.Multiply(learningRate * Errors[i], PreviousLayer.NeuronsAF));
                Biases[i] += learningRate * Errors[i];
            });
            PreviousLayer.Train(learningRate);
        }
    }
}
