using MachineLearningLib.NeuralNetwork;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLLTesterCMD
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //XORExample.XORMain();
            DigitClassifier.DCMain();

            Console.ReadKey();
        }
    }
}
