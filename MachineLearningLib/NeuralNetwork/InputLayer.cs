using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningLib.NeuralNetwork
{
    public class InputLayer : Layer
    {
        public InputLayer(int neurons) : base(neurons)
        {
        }

        public void SetInput(float[] input)
        {
            if (input.Length != NeuronsSum.Length)
                throw new InvalidOperationException("Wrong amount of inputs!");
            NeuronsAF = input;
        }

        public override void InitFromPreviousLayer()
        {
            
        }

        public override void Calculate()
        {
            FollowingLayer.Calculate();
        }

        public override void Train(float learningRate)
        {
            
        }
    }
}
