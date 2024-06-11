using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningLib
{
    public interface IActivationFunction
    {
        float Evaluate(float x);
        float Derivative(float x);
    }
}
