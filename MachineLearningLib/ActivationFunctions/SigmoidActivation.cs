using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningLib.ActivationFunctions
{
    public class SigmoidActivation : IActivationFunction
    {
        public float Derivative(float x)
        {
            var sigmoid = Evaluate(x);
            return sigmoid * (1 - sigmoid);
        }

        public float Evaluate(float x)
        {
            return 1 / (1 + (float)Math.Exp(-x));
        }
    }
}
