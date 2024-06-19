using MachineLearningLib.ActivationFunctions;
using System;
using System.Collections.Generic;
using System.IO;
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

        public void Save(Stream stream)
        {
            BinaryWriter bw = new BinaryWriter(stream);
            Layer NextLayer = inputLayer;
            while (NextLayer != null)
            {
                NextLayer.Save(bw);
                NextLayer = NextLayer.FollowingLayer;
            }
            bw.Dispose();
        }

        public void Load(Stream stream)
        {
            BinaryReader br = new BinaryReader(stream);
            Layer NextLayer = inputLayer;
            while (NextLayer != null)
            {
                NextLayer.Load(br);
                NextLayer = NextLayer.FollowingLayer;
            }
            br.Dispose();
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
            IWeightInitializer currentWI = null;
            IAccelerator currentAC = null;
            IOptimizer currentOP = null;
            Parallelizer currentPL = null;

            public Builder(NetworkHolder nh)
            {
                this.nh = nh;
            }

            public Builder Use(IActivationFunction activationFunction)
            {
                currentAF = activationFunction;
                return this;
            }

            public Builder Use(IAccelerator accelerator)
            {
                currentAC = accelerator;
                return this;
            }

            public Builder Use(Parallelizer parallelizer)
            {
                currentPL = parallelizer;
                return this;
            }

            public Builder Use(IWeightInitializer weightInitializer)
            {
                currentWI = weightInitializer;
                return this;
            }

            public Builder Use(IOptimizer optimizer)
            {
                currentOP = optimizer;
                return this;
            }

            public Builder Stack(Layer layer)
            {
                if (currentAF != null)
                    layer.ActivationFunction = currentAF;
                if (currentAC != null)
                    layer.Accelerator = currentAC;
                if (currentPL != null)
                    layer.Parallelizer = currentPL;
                if (currentWI != null)
                    layer.WeightInitializer = currentWI;
                if (currentOP != null)
                    layer.Optimizer = currentOP;

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
