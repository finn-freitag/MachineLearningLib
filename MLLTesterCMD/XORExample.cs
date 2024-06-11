using MachineLearningLib.NeuralNetwork;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLLTesterCMD
{
    public static class XORExample
    {
        public static void XORMain()
        {
            NetworkHolder nh = NetworkHolder.Create()
                .Stack(new InputLayer(2))
                .Stack(new Layer(2))
                .Stack(new OutputLayer(1))
                .Build();

            float[][] inputs = new float[][]
            {
                new float[] { 0f, 0f },
                new float[] { 0f, 1f },
                new float[] { 1f, 0f },
                new float[] { 1f, 1f }
            };
            float[][] outputs = new float[][]
            {
                new float[] { 0f },
                new float[] { 1f },
                new float[] { 1f },
                new float[] { 0f }
            };

            Console.WriteLine("XOR Training:");
            Console.WriteLine();

            int epochs = 3;
            float learningRate = 0.1f;
            for (int i = 0; i < 10000; i++)
            {
                for (int j = 0; j < inputs.GetLength(0); j++)
                    nh.Train(inputs[j], outputs[j], epochs, learningRate);
                Console.WriteLine("Epoch: " + (i + 1));
            }

            Console.WriteLine();

            for (int j = 0; j < inputs.GetLength(0); j++)
            {
                var res = nh.Calculate(inputs[j]);
                Console.WriteLine("[" + string.Join(",", inputs[j]) + "] => " + (res[0] + "").Replace(",", "."));
            }

            Console.ReadKey();
        }
    }
}
