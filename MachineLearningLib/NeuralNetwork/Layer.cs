using MachineLearningLib.Accelerators;
using MachineLearningLib.ActivationFunctions;
using MachineLearningLib.Parallelizers;
using MachineLearningLib.WeightInitializers;
using System;
using System.Collections.Generic;
using System.IO;
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

        public IActivationFunction ActivationFunction = new SigmoidActivation();
        public IWeightInitializer WeightInitializer = new RandomWeightInitializer();
        public IAccelerator Accelerator = new NoAccelerator();
        public Parallelizer Parallelizer;

        public object Tag;

        public Layer(int neurons)
        {
            Biases = new float[neurons];
            NeuronsSum = new float[neurons];
            NeuronsAF = new float[neurons];
            Errors = new float[neurons];
            Parallelizer = NoParallelizer.Parallelizer;
        }

        public virtual void InitFromPreviousLayer()
        {
            WeightInitializer.SetLayer(this);
            Weights = new float[NeuronsSum.Length][];
            Parallelizer(0, NeuronsSum.Length, (i) =>
            {
                Weights[i] = new float[PreviousLayer.NeuronsSum.Length];
                Biases[i] = WeightInitializer.GetInitialWeight();
                for (int j = 0; j < PreviousLayer.NeuronsSum.Length; j++)
                {
                    Weights[i][j] = WeightInitializer.GetInitialWeight();
                }
            });
        }

        public virtual void Calculate()
        {
            Parallelizer(0, NeuronsSum.Length, (i) =>
            {
                NeuronsSum[i] = Accelerator.DotProduct(PreviousLayer.NeuronsAF, Weights[i]);
                NeuronsSum[i] += Biases[i];
                NeuronsAF[i] = ActivationFunction.Evaluate(NeuronsSum[i]);
            });
            FollowingLayer.Calculate();
        }

        public virtual void Train(float learningRate)
        {
            Parallelizer(0, NeuronsSum.Length, (i) =>
            {
                Errors[i] = Accelerator.DotProductT(FollowingLayer.Errors, FollowingLayer.Weights, i);
                Errors[i] *= ActivationFunction.Derivative(NeuronsSum[i]);
                Weights[i] = Accelerator.Add(Weights[i], Accelerator.Multiply(learningRate * Errors[i], PreviousLayer.NeuronsAF));
                Biases[i] += learningRate * Errors[i];
            });
            PreviousLayer.Train(learningRate);
        }

        public virtual void Save(BinaryWriter bw)
        {
            bw.Write(Biases.Length);
            for (int i = 0; i < Biases.Length; i++)
            {
                bw.Write((double)Biases[i]);
            }
            bw.Write(Weights.Length);
            for (int i = 0; i < Weights.Length; i++)
            {
                bw.Write(Weights[i].Length);
                for (int j = 0; j < Weights[i].Length; j++)
                {
                    bw.Write((double)Weights[i][j]);
                }
            }
        }

        public virtual void Load(BinaryReader br)
        {
            int length = br.ReadInt32();
            if (length != Biases.Length)
                throw new InvalidOperationException("Weight data isn't made for this network!");
            for (int i = 0; i < length; i++)
            {
                Biases[i] = (float)br.ReadDouble();
            }
            length = br.ReadInt32();
            if (length != Weights.Length)
                throw new InvalidOperationException("Weight data isn't made for this network!");
            for (int i = 0; i < length; i++)
            {
                int length2 = br.ReadInt32();
                if (length2 != Weights[i].Length)
                    throw new InvalidOperationException("Weight data isn't made for this network!");
                for (int j = 0; j < length2; j++)
                {
                    Weights[i][j] = (float)br.ReadDouble();
                }
            }
        }
    }
}
