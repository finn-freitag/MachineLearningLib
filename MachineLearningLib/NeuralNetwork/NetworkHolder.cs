using MachineLearningLib.ActivationFunctions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningLib.NeuralNetwork
{
    public class NetworkHolder
    {
        InputLayer inputLayer;
        OutputLayer outputLayer;

        private NetworkHolder()
        {

        }

        public static Builder Create()
        {
            return new Builder(new NetworkHolder());
        }

        public float[] Calculate(float[] inputs)
        {
            inputLayer.SetInput(inputs);
            inputLayer.Calculate();
            return outputLayer.GetOutputs();
        }

        public void Train(float[] inputs, float[] desiredOutputs, int epochs = 3, float learningRate = 0.01f)
        {
            for(int i = 0; i < epochs; i++)
            {
                Calculate(inputs);
                outputLayer.SetDesiredOutputs(desiredOutputs);
                outputLayer.Train(learningRate);
            }
        }

        public class Builder
        {
            NetworkHolder nh;
            List<Layer> layers = new List<Layer>();

            IActivationFunction currentAF = null;

            public Builder(NetworkHolder nh)
            {
                this.nh = nh;
            }

            public Builder Use(IActivationFunction activationFunction)
            {
                currentAF = activationFunction;
                return this;
            }

            public Builder Stack(Layer layer)
            {
                if (currentAF != null)
                    layer.ActivationFunction = currentAF;

                layers.Add(layer);

                return this;
            }

            public NetworkHolder Build()
            {
                if (!(layers[0] is InputLayer))
                    throw new InvalidOperationException("First layer needs to be an input layer!");
                if (!(layers[layers.Count - 1] is OutputLayer))
                    throw new InvalidOperationException("Last layer needs to be an output layer!");

                for(int i = 1; i < layers.Count; i++)
                {
                    layers[i - 1].FollowingLayer = layers[i];
                    layers[i].PreviousLayer = layers[i - 1];
                }

                for (int i = 0; i < layers.Count; i++)
                    layers[i].InitFromPreviousLayer();

                nh.inputLayer = (InputLayer)layers[0];
                nh.outputLayer = (OutputLayer)layers[layers.Count - 1];

                return nh;
            }
        }
    }
}
