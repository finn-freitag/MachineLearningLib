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
            Random r = new Random();
            Weights = new float[NeuronsSum.Length][];
            for(int i = 0; i < NeuronsSum.Length; i++)
            {
                Weights[i] = new float[PreviousLayer.NeuronsSum.Length];
                Biases[i] = (float)r.NextDouble();
                for (int j = 0; j < PreviousLayer.NeuronsSum.Length; j++)
                {
                    Weights[i][j] = (float)r.NextDouble();
                }
            }
        }

        public virtual void Calculate()
        {
            var sigmoid = new SigmoidActivation();
            for(int i = 0; i < NeuronsSum.Length; i++)
            {
                float sum = 0f;
                for(int j = 0; j < PreviousLayer.NeuronsSum.Length; j++)
                {
                    sum += Weights[i][j] * PreviousLayer.NeuronsAF[j];
                }
                sum += Biases[i];
                NeuronsSum[i] = sum;
                NeuronsAF[i] = sigmoid.Evaluate(sum);
            }
            FollowingLayer.Calculate();
        }

        public virtual void Train(float learningRate)
        {
            var sigmoid = new SigmoidActivation();
            for (int i = 0; i < NeuronsSum.Length; i++)
            {
                float error = 0f;
                for(int j = 0; j < FollowingLayer.Errors.Length; j++)
                {
                    error += FollowingLayer.Errors[j] * FollowingLayer.Weights[j][i];
                }
                Errors[i] = error * sigmoid.Derivative(NeuronsSum[i]);
                for(int j = 0; j < PreviousLayer.NeuronsAF.Length; j++)
                {
                    Weights[i][j] += learningRate * Errors[i] * PreviousLayer.NeuronsAF[j];
                }
                Biases[i] += learningRate * Errors[i];
            }
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
