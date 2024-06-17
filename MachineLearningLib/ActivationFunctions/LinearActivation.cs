using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningLib.ActivationFunctions
{
    public class LinearActivation : IActivationFunction
    {
        public float Derivative(float x)
        {
            return 1;
        }

        public float Evaluate(float x)
        {
            return x;
        }
    }
}
