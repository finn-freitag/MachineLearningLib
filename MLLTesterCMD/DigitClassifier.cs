using System;
using System.Collections.Generic;
using System.Drawing;
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
            string weightPath = "weights.dat";
            string test = "testImages";

            Console.WriteLine("Digit Classifier:");
            Console.WriteLine();

            Console.WriteLine("Loading weights...");
            DigitRecognition recognizer = new DigitRecognition(28, 28);
            recognizer.Load(new MemoryStream(File.ReadAllBytes(weightPath)));

            foreach(string file in Directory.GetFiles(test))
            {
                Bitmap bmp = new Bitmap(file);
                DigitData data = DigitClassification.ImageConverter.ToDigitData(bmp);
                var res = recognizer.Classify(data);
                Console.WriteLine("File: " + Path.GetFileNameWithoutExtension(file));
                for(int i = 0; i < res.Length; i++)
                {
                    Console.WriteLine(i + ": " + res[i]);
                }
                Console.WriteLine("Digit: " + data.Digit);
                Console.WriteLine();
            }
        }
    }
}
