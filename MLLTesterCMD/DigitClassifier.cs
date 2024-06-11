using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLLTesterCMD
{
    public static class DigitClassifier
    {
        public static void DCMain()
        {
            string imagePath = "mnist\\t10k-images.idx3-ubyte";
            string labelPath = "mnist\\t10k-labels.idx1-ubyte";
            string weightPath = "weights.dat";

            Console.WriteLine("Digit classifier:");
            Console.WriteLine();

            Console.WriteLine("Loading dataset...");

            DigitData[] data = MNistLoader.LoadFromFile(imagePath, labelPath);

            DigitRecognition recognizer = new DigitRecognition(data[0].Width, data[0].Height);
            recognizer.Train(data, 10, 0.1f);
            //recognizer.Load(new MemoryStream(File.ReadAllBytes(weightPath)));

            Console.WriteLine();
            Console.WriteLine("Training done.");
            Console.WriteLine("Saving weights...");
            MemoryStream ms = new MemoryStream();
            recognizer.Save(ms);
            File.WriteAllBytes(weightPath, ms.ToArray());
            Console.WriteLine("Started classification...");

            int[] digits = new int[data.Length];
            for (int i = 0; i < data.Length; i++)
            {
                digits[i] = data[i].Digit;
                data[i].Digit = -1;
            }

            recognizer.Classify(data);

            int correct = 0;
            for(int i = 0; i < digits.Length; i++)
            {
                if (data[i].Digit == digits[i])
                    correct++;
            }

            Console.WriteLine("Classification done: " + correct + "/" + digits.Length);
        }
    }
}
