using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningLib.ActivationFunctions
{
    public class LeakyReLU : IActivationFunction
    {
        float alpha = 0.1f;

        public LeakyReLU(float alpha = 0.01f)
        {
            this.alpha = alpha;
        }

        public float Derivative(float x)
        {
            if (x > 0)
                return 1;
            else
                return alpha;
        }

        public float Evaluate(float x)
        {
            if (x > 0)
                return x;
            else
                return alpha * x;
        }
    }
}
