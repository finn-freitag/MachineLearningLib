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
    public abstract class Layer
    {
        public float[] Biases;
        public float[] NeuronsSum;
        public float[] NeuronsAF;
        public float[] Errors;
        public float[][] Weights;

        public Layer FollowingLayer;
        public Layer PreviousLayer;

        public object Tag;

        public Layer(int neurons)
        {
            Biases = new float[neurons];
            NeuronsSum = new float[neurons];
            NeuronsAF = new float[neurons];
            Errors = new float[neurons];
        }

        public abstract void InitFromPreviousLayer();

        public abstract void Calculate();

        public abstract void Train(float learningRate);

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
