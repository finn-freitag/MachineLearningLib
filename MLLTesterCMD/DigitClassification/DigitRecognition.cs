using MachineLearningLib.NeuralNetwork;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLLTesterCMD
{
    public class DigitRecognition
    {
        NetworkHolder network;

        public DigitRecognition(int imageWidth, int imageHeight)
        {
            network = NetworkHolder.Create()
                .Stack(new InputLayer(imageWidth * imageHeight))
                .Stack(new Layer(128))
                .Stack(new Layer(64))
                .Stack(new OutputLayer(10))
                .Build();
        }

        public void Train(DigitData[] data, int epochs, float learningRate = 0.01f)
        {
            for(int j = 0; j < epochs; j++)
            {
                for (int i = 0; i < data.Length; i++)
                {
                    network.Train(Prepare(data[i].Data), GetDigitArrayFromDigit(data[i].Digit), 2, learningRate);
                    if (i % 1000 == 0)
                        Console.WriteLine("Epoch: " + j + "/" + epochs + ", Cycle: " + i + "/" + data.Length);
                }
            }
        }

        public void Classify(DigitData[] data)
        {
            for(int i = 0; i < data.Length; i++)
            {
                data[i].Digit = GetDigitFromDigitArray(network.Calculate(Prepare(data[i].Data)));
            }
        }

        public void Classify(DigitData data)
        {
            data.Digit = GetDigitFromDigitArray(network.Calculate(Prepare(data.Data)));
        }

        public void Save(Stream stream) => network.Save(stream);
        public void Load(Stream stream) => network.Load(stream);

        private float[] Prepare(byte[] bytes)
        {
            float[] res = new float[bytes.Length];
            for(int i = 0; i < bytes.Length; i++)
            {
                res[i] = bytes[i] / 255.0f;
            }
            return res;
        }

        private float[] GetDigitArrayFromDigit(int digit)
        {
            float[] res = new float[10];
            for(int i = 0; i < res.Length; i++)
            {
                if (i == digit)
                    res[i] = 1;
                else
                    res[i] = 0;
            }
            return res;
        }

        private int GetDigitFromDigitArray(float[] d)
        {
            int maxIndex = 0;
            float max = d[0];
            for(int i = 1; i < d.Length; i++)
            {
                if(max < d[i])
                {
                    maxIndex = i;
                    max = d[i];
                }
            }
            return maxIndex;
        }
    }
}
