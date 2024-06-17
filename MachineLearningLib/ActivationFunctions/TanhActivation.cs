using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningLib.ActivationFunctions
{
    public class TanhActivation : IActivationFunction
    {
        public float Derivative(float x)
        {
            float eval = Evaluate(x);
            return 1 - eval * eval;
        }

        public float Evaluate(float x)
        {
            return (float)((Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x)));
        }
    }
}
