using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningLib.ActivationFunctions
{
    public class ReLUActivation : IActivationFunction
    {
        public float Derivative(float x)
        {
            if (x < 0)
                return 0;
            else
                return 1;
        }

        public float Evaluate(float x)
        {
            return Math.Max(0, x);
        }
    }
}
