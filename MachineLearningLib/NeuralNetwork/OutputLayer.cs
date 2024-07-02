using MachineLearningLib.ActivationFunctions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningLib.NeuralNetwork
{
    public abstract class OutputLayer : Layer
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

        public abstract override void Calculate();

        public abstract override void Train(float learningRate);
    }
}
